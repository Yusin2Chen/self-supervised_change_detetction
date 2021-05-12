from __future__ import print_function

import os
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torch.utils.data import DataLoader
from data.dataset_s2 import DFC2020
from models.resnet import ResNet18, Net6
import utils.metrics as metrics
from utils.estimators import HardNegtive_loss, tuba_lower_bound, smile_lower_bound, infonce_lower_bound, dv_upper_lower_bound
# data augmentation
from kornia import geometry as geo
from kornia import filters
from skimage import filters
from utils.util import adjust_learning_rate, AverageMeter, rosin, GaussianBlur, default, RandomApply



def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    # load the original big image (if just one, it should be big enough)
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size for data training')
    parser.add_argument('--crop_size', type=int, default=200, help='crop_size for ensuring same patch_size within all batches')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    # split image into small patches
    parser.add_argument('--patch_size', type=int, default=8, help='patch_size for training')
    parser.add_argument('--unfold_stride', type=int, default=4, help='stride during the training patches')
    parser.add_argument('--val_patch_size', type=int, default=8, help='patch_size for inference')
    parser.add_argument('--val_unfold_stride', type=int, default=2, help='stride during the inference patches')
    parser.add_argument('--pbatch_size', type=int, default=10000, help='batch_size of patches during inference')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    # resume and test
    parser.add_argument('--resume', default=False, type=bool, help='flag for training from checkpoint')
    parser.add_argument('--test', default=False, type=bool, help='flag for testing on test data set')
    # model definition
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34'])
    parser.add_argument('--feat_dim', type=int, default=64, help='dim of feat for inner product')
    # input/output for data use
    parser.add_argument('--use_s2hr', action='store_true', default=True, help='use sentinel-2 high-resolution (10 m) bands')
    parser.add_argument('--use_s2mr', action='store_true', default=False, help='use sentinel-2 medium-resolution (20 m) bands')
    parser.add_argument('--use_s2lr', action='store_true', default=False, help='use sentinel-2 low-resolution (60 m) bands')
    parser.add_argument('--use_s1', action='store_true', default=True, help='use sentinel-1 data')
    parser.add_argument('--no_savanna', action='store_true', default=False, help='ignore class savanna')
    # specify folder
    parser.add_argument('--data_dir_train', type=str, default='./InferS2-all', help='path to training data set')
    parser.add_argument('--data_dir_eval', type=str, default='./InferS2', help='path to test data set')
    parser.add_argument('--save_path', type=str, default='./save_MIE', help='path to save model')
    parser.add_argument('--eval_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
    opt = parser.parse_args()

    if (opt.data_dir_train is None):
        raise ValueError('one or more of the folders is None: data_folder')
    opt.model_name = opt.model
    opt.model_name = 'calibrated_{}_bsz_{}_lr_{}_decay_{}'.format(opt.model_name, opt.batch_size, opt.learning_rate,
                                                                  opt.weight_decay)
    opt.save_path = os.path.join(opt.save_path, opt.model_name)
    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)
    if not os.path.isdir(opt.data_dir_train):
        raise ValueError('data path not exist: {}'.format(opt.data_dir_train))
    return opt

def change_map(difference_img):
    #difference_img = difference_img.cpu().detach().numpy()
    #threshold = filters.threshold_otsu(difference_img)
    #rosin methods
    difference_img = (difference_img - difference_img.mean()) / difference_img.std()
    #threshold = rosin(difference_img)
    threshold = difference_img.min().abs()

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
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    eval_loader = DataLoader(eval_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=False)

    return train_loader, eval_loader, n_inputs, n_classes

class Trainer:
    def __init__(self, args, online_network, target_network, optimizer, device):

        #DEFAULT_AUG = nn.Sequential(
        #    RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.5),
        #    RandomApply(geo.transform.Hflip(), p=0.5),
        #    RandomApply(geo.transform.Vflip(), p=0.5))
        #augment_fn = None
        #self.augment = default(augment_fn, DEFAULT_AUG)
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.savepath = args.save_path
        self.max_epochs = args.epochs
        self.m = 0.996
        self.n_classes = args.n_classes
        self.patch_size = args.patch_size
        self.pbatch_size = args.pbatch_size
        self.train_pbatch_size = int(args.pbatch_size * 0.5)
        self.unfold_stride = args.unfold_stride
        self.val_patch_size = args.val_patch_size
        self.val_unfold_stride = args.val_unfold_stride
        self.eval_freq = args.eval_freq
        self.save_freq = args.save_freq

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_loader, eval_loader):

        niter = 0

        for epoch_counter in range(self.max_epochs):
            train_loss = 0.0
            for idx, (batch, _) in enumerate(train_loader):
                image = batch['image']
                # split whole image to patches
                patches = self.patchize(image, self.patch_size, self.unfold_stride)
                P, C, pH, pW = patches.shape
                # random shuffle index
                shuffle_ids = torch.randperm(P).cuda()
                # shuffle for training
                this_patches = patches[shuffle_ids]
                # training in each pbatch_size
                quotient, remainder = divmod(P, self.train_pbatch_size)
                pbatch = quotient if quotient > 0 else remainder
                for i in range(pbatch):
                    start = i * self.train_pbatch_size
                    end = start + self.train_pbatch_size

                    patch = this_patches[start:end, :, :, :]
                    # read file
                    loss = self.update(patch)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    niter += 1
                    train_loss += loss.item()

                train_loss = train_loss / self.pbatch_size
                print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch_counter, train_loss))
                torch.cuda.empty_cache()

            if (epoch_counter + 1) % self.eval_freq == 0:
                self.validate(eval_loader)
                self.online_network.train()

            # save checkpoints
            if (epoch_counter + 1) % self.save_freq == 0:
                self.save_model(os.path.join(self.savepath,
                                             'MIE_epoch_{epoch}_{loss}.pth'.format(epoch=epoch_counter, loss=train_loss)))
    def update(self, image):
        # split pre and post
        batch_view_1, batch_view_2 = torch.split(image, [4, 4], dim=1)
        # if you want to train your network on single image
        #image1, _ = torch.split(image, [4, 4], dim=1)
        #batch_view_1, batch_view_2 = self.augment(image1), self.augment(image1)
        batch_view_1 = batch_view_1.to(self.device)
        batch_view_2 = batch_view_2.to(self.device)
        # compute query feature
        o_feature1 = self.online_network(batch_view_1)
        o_feature2 = self.online_network(batch_view_2)

        # compute key features
        #with torch.no_grad():
        #    t_feature2 = self.target_network(batch_view_1)
        #    t_feature1 = self.target_network(batch_view_2)

        # loss function options (normal contrastive loss)
        l_feature1 = F.normalize(o_feature1, dim=1)
        l_feature2 = F.normalize(o_feature2, dim=1)
        scores = torch.matmul(l_feature1, l_feature2.t())
        mi_estimation = smile_lower_bound(scores)
        loss = - mi_estimation
        # hard negative loss (it seems that using mean teacher in small data set leads to performance drops)
        #loss = HardNegtive_loss(o_feature1, t_feature1, batch_size=self.train_pbatch_size) +  \
        #       HardNegtive_loss(o_feature2, t_feature2, batch_size=self.train_pbatch_size)
        #loss = HardNegtive_loss(o_feature1, o_feature2, batch_size=self.train_pbatch_size)
        return loss

    def validate(self, val_loader):

        # switch to evaluate mode
        self.online_network.eval()

        # main validation loop
        conf_mat = metrics.ConfMatrix(self.n_classes, 100)

        with torch.no_grad():

            for idx, (batch, _) in enumerate(val_loader):
                start = time.time()
                # unpack sample
                image, target = batch['image'], batch['label']
                # ===================forward=====================
                prediction = self.compute_heatmap(image, self.val_patch_size, self.val_unfold_stride)
                print('time elapsed:', time.time() - start)
                cd_map = change_map(prediction)
                plt.imsave('MIEI.png', prediction.squeeze().cpu().detach().numpy(), cmap='gray')
                plt.imsave('MIE.png', np.squeeze(cd_map), cmap='gray')
                # calculate error metrics
                conf_mat.add_batch(target.cpu().numpy(), np.expand_dims(cd_map, axis=0))

                print("[Val] AA: {:.2f}%".format(conf_mat.get_aa() * 100))


    def patchize(self, img: torch.Tensor, patch_size, unfold_stride) -> torch.Tensor:

        """
        img.shape
        B  : batch size
        C  : channels of image (same to patches.shape[1])
        iH : height of image
        iW : width of image

        pH : height of patch
        pW : width of patch
        V  : values in a patch (pH * pW * C)
        """

        B, C, iH, iW = img.shape
        pH = patch_size
        pW = patch_size

        unfold = nn.Unfold(kernel_size=(pH, pW), stride=unfold_stride)

        patches = unfold(img)  # (B, V, P)
        patches = patches.permute(0, 2, 1).contiguous()  # (B, P, V)
        patches = patches.view(-1, C, pH, pW)  # (P, C, pH, pW)
        return patches

    def compute_squared_l2_distance(self, pred: torch.Tensor, surrogate_label: torch.Tensor) -> torch.Tensor:

        losses = (pred - surrogate_label) ** 2
        losses = losses.view(losses.shape[0], -1)
        losses = torch.mean(losses, dim=1)
        losses = losses.cpu().detach()

        return losses

    def compute_heatmap(self, img: torch.Tensor, patch_size, unfold_stride):

        """
        img.shape -> (B, C, iH, iW)
        B  : batch size
        C  : channels of image (same to patches.shape[1])
        iH : height of image
        iW : width of image

        patches.shape -> (P, C, pH, pW)
        P  : patch size
        C  : channels of image (same to img.shape[1])
        pH : height of patch
        pW : width of patch
        """

        patches = self.patchize(img, patch_size, unfold_stride)

        B, C, iH, iW = img.shape
        P, C, pH, pW = patches.shape

        heatmap = torch.zeros(P)
        quotient, remainder = divmod(P, self.pbatch_size)

        for i in range(quotient):

            start = i * self.pbatch_size
            end = start + self.pbatch_size

            patch = patches[start:end, :, :, :]
            patch = patch.to(self.device)

            # same network
            patch1, patch2 = torch.split(patch, [4, 4], dim=1)
            surrogate_label = self.online_network(patch1)
            pred = self.online_network(patch2)
            #
            losses = self.compute_squared_l2_distance(pred, surrogate_label)
            heatmap[start:end] = losses

        if remainder != 0:
            patch = patches[-remainder:, :, :, :]
            patch = patch.to(self.device)
            patch1, patch2 = torch.split(patch, [4, 4], dim=1)
            surrogate_label = self.online_network(patch1)
            pred = self.online_network(patch2)
            #
            losses = self.compute_squared_l2_distance(pred, surrogate_label)
            heatmap[-remainder:] = losses

        fold = nn.Fold(
            output_size=(iH, iW),
            kernel_size=(pH, pW),
            stride=unfold_stride,
        )

        heatmap = heatmap.expand(B, pH * pW, P)
        heatmap = fold(heatmap)
        heatmap = heatmap.squeeze()

        del patches
        return heatmap

    def save_model(self, PATH):
        print('==> Saving...')
        state ={
            'online_network_state_dict': self.online_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(state, PATH)
        # help release GPU memory
        del state


def main():

    # parse the args
    args = parse_option()

    # set flags for GPU processing if available
    if torch.cuda.is_available():
        args.use_gpu = True
        device = 'cuda'
    else:
        args.use_gpu = False
        device = 'cpu'

    # set the data loader
    train_loader, eval_loader, n_inputs, n_classes = get_train_val_loader(args)
    args.n_inputs = n_inputs
    args.n_classes = 2

    # set the model
    # online_network = ResNet18(num_feats=args.feat_dim, width=1, in_channel=4).to(device)
    online_network = Net6(num_feats=args.feat_dim, width=1, in_channel=4).to(device)
    # --> target model
    target_network = copy.deepcopy(online_network)
    target_network = target_network.to(device)

    if args.resume:
        try:
            print('loading pretrained models')
            checkpoints_folder = os.path.join('.', 'pre_train')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'MIE' + str(args.crop_size) + '.pth')),
                                     map_location=device)
            online_network.load_state_dict(load_params['online_network_state_dict'])

            if args.test:
                trainer = Trainer(args, online_network=online_network, target_network=None, optimizer=None, device=device)

                trainer.validate(train_loader)

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # target encoder
    optimizer = torch.optim.SGD(online_network.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0004)

    trainer = Trainer(args,
                        online_network=online_network,
                        target_network=target_network,
                        optimizer=optimizer,
                        device=device)

    trainer.train(train_loader, eval_loader)



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
