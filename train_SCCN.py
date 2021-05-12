from __future__ import print_function

import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torch.utils.data import DataLoader
from skimage import filters
from utils.util import adjust_learning_rate, AverageMeter, patchize, rosin
from data.dataset_s2 import DFC2020
from models.networks import SCCN
import utils.metrics as metrics

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    # read one big image
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--crop_size', type=int, default=200, help='crop_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    # split image into small patches
    parser.add_argument('--patch_size', type=int, default=100, help='patch_size')
    parser.add_argument('--pbatch_size', type=int, default=2, help='batch_size of patches')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    # resume path
    parser.add_argument('--resume', default=False, type=bool, help='flag for training from checkpoint')
    parser.add_argument('--test', default=False, type=bool, help='flag for testing on test data set')
    # model definition
    parser.add_argument('--model', type=str, default='SCCN')
    # input/output
    parser.add_argument('--use_s2hr', action='store_true', default=True, help='use sentinel-2 high-resolution (10 m) bands')
    parser.add_argument('--use_s2mr', action='store_true', default=False, help='use sentinel-2 medium-resolution (20 m) bands')
    parser.add_argument('--use_s2lr', action='store_true', default=False, help='use sentinel-2 low-resolution (60 m) bands')
    parser.add_argument('--use_s1', action='store_true', default=True, help='use sentinel-1 data')
    parser.add_argument('--no_savanna', action='store_true', default=False, help='ignore class savanna')
    # specify folder
    parser.add_argument('--data_dir_train', type=str, default='./InferS2-all', help='path to training dataset')
    parser.add_argument('--data_dir_eval', type=str, default='./InferS2', help='path to training dataset')
    parser.add_argument('--save_path', type=str, default='./save_SCCN', help='path to save linear classifier')
    parser.add_argument('--eval_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=200, help='save frequency')
    opt = parser.parse_args()
    if (opt.data_dir_train is None):
        raise ValueError('one or more of the folders is None: data_folder')
    opt.model_name = opt.model
    opt.model_name = 'calibrated_{}_bsz_{}_lr_{}_decay_{}'.format(opt.model_name, opt.batch_size, opt.learning_rate,
                                                                  opt.weight_decay)
    opt.save_folder = os.path.join(opt.save_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    if not os.path.isdir(opt.data_dir_train):
        raise ValueError('data path not exist: {}'.format(opt.data_dir_train))
    return opt

def weighted_mse_loss(input, target, weights=None):
    if weights is None:
        out = (input - target)**2
        loss = out.mean()
        return loss
    else:
        out = (input - target)**2
        out = out * weights.expand_as(out)
        loss = out.mean()
        return loss

def change_map(difference_img):

    difference_img = difference_img.cpu().detach().numpy()
    threshold = filters.threshold_otsu(difference_img)
    # rosin methods
    #difference_img = (difference_img - difference_img.mean()) / difference_img.std()
    #threshold = rosin(difference_img)
    #threshold = 2 * abs(min(difference_img))
    return difference_img >= threshold

def get_train_val_loader(args):
    data_set = DFC2020(args.data_dir_train,
                    subset="train",
                    no_savanna=args.no_savanna,
                    use_s2hr=args.use_s2hr,
                    use_s2mr=args.use_s2mr,
                    use_s2lr=args.use_s2lr,
                    use_s1=args.use_s1,
                    unlabeled=True,
                    transform=True,
                    train_index=None,
                    crop_size=args.crop_size)
    n_classes = data_set.n_classes
    n_inputs = data_set.n_inputs

    eval_set = DFC2020(args.data_dir_eval,
                       subset="train",
                       no_savanna=args.no_savanna,
                       use_s2hr=args.use_s2hr,
                       use_s2mr=args.use_s2mr,
                       use_s2lr=args.use_s2lr,
                       use_s1=args.use_s1,
                       unlabeled=False,
                       transform=False,
                       train_index=None,
                       crop_size=args.crop_size)

    # set up dataloaders
    train_loader = DataLoader(data_set,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    eval_loader = DataLoader(eval_set,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=False)

    return train_loader, eval_loader, n_inputs, n_classes

def set_model(args):
    if args.model.startswith('SCCN'):
        model = SCCN(4, 4).to(args.device)
    else:
        raise NotImplementedError('model not supported {}'.format(args.model))

    criterion = weighted_mse_loss
    return model, criterion


def pretrain(epoch, train_loader, classifier, criterion, optimizer, args):
    """
    one epoch training
    """
    # set model to train mode
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    loss = None

    for idx, (batch, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # unpack sample
        image = batch['image']
        # split whole image to patches
        patches = patchize(image, args.patch_size, args.patch_size)
        # B, C, iH, iW = image.shape
        P, C, pH, pW = patches.shape
        quotient, remainder = divmod(P, args.pbatch_size)
        pbatch = quotient if quotient > 0 else remainder
        for i in range(pbatch):
            start = i * args.pbatch_size
            end = start + args.pbatch_size

            patch = patches[start:end, :, :, :]
            patch = patch.to(args.device)
            # read file
            x, y = torch.split(patch, [4, 4], dim=1)
            # ===================forward=====================
            x_tilde, y_tilde = classifier(x, y, training=True, pretraining=True)
            recon_x_loss = criterion(x, x_tilde)
            recon_y_loss = criterion(y, y_tilde)
            loss = recon_x_loss + recon_y_loss
            # ===================backward=====================
            # reset gradients
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()
        loss_epoch = loss
        # print info
        print(f'\rtrain loss : {loss_epoch.item():.5f}| step :{idx}/{len(train_loader)}|{epoch}', end='', flush=True)


def train(epoch, train_loader, eval_loader, classifier, criterion, optimizer, args):
    """
    one epoch training
    """
    # set model to train mode
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    loss = None

    for idx, (batch, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # unpack sample
        image = batch['image']
        # read file
        if epoch > 25:
            with torch.no_grad():
                image_test = batch['image'].to(args.device)
                x, y = torch.split(image_test, [4, 4], dim=1)
                pseudo_cd = classifier(x, y, training=False)
                pseudo_cd = change_map(pseudo_cd)
                clw = 1.0 - pseudo_cd
                pri_clw = clw
            pri_clw = torch.unsqueeze(torch.from_numpy(pri_clw), dim=0)
            image = torch.cat((image, pri_clw.to(torch.float32)), dim=1)
        else:
            Pu = torch.ones(1, 1, image.shape[2], image.shape[3])
            image = torch.cat((image, Pu), dim=1)
        # split whole image to patches
        patches = patchize(image, args.patch_size, args.patch_size)
        P, C, pH, pW = patches.shape
        quotient, remainder = divmod(P, args.pbatch_size)
        pbatch = quotient if quotient > 0 else remainder
        for i in range(pbatch):
            start = i * args.pbatch_size
            end = start + args.pbatch_size

            patch = patches[start:end, :, :, :]
            patch = patch.to(args.device)
            # read file
            x, y, clw = torch.split(patch, [4, 4, 1], dim=1)
            # ===================forward=====================
            x_code, y_code = classifier(x, y, training=True)
            # calculate the loss with weight
            loss = criterion(x_code, y_code, clw)
            loss = loss - 1 * torch.mean(clw)
            # ===================backward=====================
            # reset gradients
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()
        loss_epoch = loss
        # print info
        print(f'\rtrain loss : {loss_epoch.item():.5f}| step :{idx}/{len(train_loader)}|{epoch}', end='', flush=True)

    # validation
    if epoch % args.eval_freq == 0:
        validate(eval_loader, classifier, args)


def validate(val_loader, classifier, args):
    """
    evaluation
    """
    # switch to evaluate mode
    classifier.eval()

    # main validation loop
    conf_mat = metrics.ConfMatrix(args.n_classes, args.crop_size)

    with torch.no_grad():
        #end = time.time()
        for idx, (batch, _) in enumerate(val_loader):

            # unpack sample
            image, target = batch['image'], batch['label']
            image = image.to(args.device)
            pre_img, pos_img = torch.split(image, [4, 4], dim=1)
            # ===================forward=====================
            start = time.time()
            prediction = classifier(pre_img, pos_img, training=False)
            print('time elapsed:', time.time() - start)
            cd_map = change_map(prediction)
            plt.imsave('SCCN.png', np.squeeze(cd_map), cmap='gray')
            # calculate error metrics
            conf_mat.add_batch(target.cpu().numpy(), cd_map)

        # close progressbar
        print("[Val] AA: {:.2f}%".format(conf_mat.get_aa() * 100))



def main():

    # parse the args
    args = parse_option()
    # set flags for GPU processing if available
    if torch.cuda.is_available():
        args.use_gpu = True
        args.device = 'cuda'
    else:
        args.use_gpu = False
        args.device = 'cpu'
    # set the data loader
    train_loader, eval_loader, n_inputs, n_classes = get_train_val_loader(args)
    args.n_inputs = n_inputs
    args.n_classes = 2
    # set the model
    classifier, criterion = set_model(args)

    if args.resume:
        try:
            print('loading pretrained models')
            checkpoints_folder = os.path.join('.', 'pre_train')
            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'SCCN.pth')), map_location=args.device)
            classifier.load_state_dict(load_params['classifier'])

            if args.test:
                validate(train_loader, classifier, args)

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # set optimizer
    # optimizer = optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    # routine
    args.start_epoch = 1
    for epoch in range(args.start_epoch, args.start_epoch + 20):
        pretrain(epoch, train_loader, classifier, criterion, optimizer, args)
    print('\n Pretraining Finishing!')
    for epoch in range(args.start_epoch, args.epochs + 1):
        #adjust_learning_rate(epoch, args, optimizer)
        train(epoch, train_loader, eval_loader, classifier, criterion, optimizer, args)
        #scheduler.step()


        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'epoch': epoch,
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_name = 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch)
            save_name = os.path.join(args.save_folder, save_name)
            print('saving regular model!')
            torch.save(state, save_name)



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
