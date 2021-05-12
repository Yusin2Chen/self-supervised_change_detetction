from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import scipy
import numbers
import random
from matplotlib import colors
import matplotlib.patches as mpatches
from statsmodels.nonparametric.kde import KDEUnivariate
from PIL import ImageFilter


from kornia import augmentation as augs
from kornia import filters, color

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def convert_to_np(tensor):
    # convert pytorch tensors to numpy arrays
    if not isinstance(tensor, np.ndarray):
        tensor = tensor.cpu().numpy()
    return tensor


def labels_to_dfc(tensor, no_savanna):
    """
    INPUT:
    Classes encoded in the training scheme (0-9 if savanna is a valid label
    or 0-8 if not). Invalid labels are marked by 255 and will not be changed.

    OUTPUT:
    Classes encoded in the DFC2020 scheme (1-10, and 255 for invalid).
    """

    # transform to numpy array
    tensor = convert_to_np(tensor)

    # copy the original input
    out = np.copy(tensor)

    # shift labels if there is no savanna class
    if no_savanna:
        for i in range(2, 9):
            out[tensor == i] = i + 1
    else:
        pass

    # transform from zero-based labels to 1-10
    out[tensor != 255] += 1

    # make sure the mask is intact and return transformed labels
    assert np.all((tensor == 255) == (out == 255))
    return out


def display_input_batch(tensor, display_indices=0, brightness_factor=3):

    # extract display channels
    tensor = tensor[:, display_indices, :, :]

    # restore NCHW tensor shape if single channel image
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)

    # scale image
    tensor = torch.clamp((tensor * brightness_factor), 0, 1)

    return tensor


def display_label_batch(tensor, no_savanna=False):

    # get predictions if input is one-hot encoded
    if len(tensor.shape) == 4:
        tensor = tensor.max(1)[1]

    # convert train labels to DFC2020 class scheme
    tensor = labels_to_dfc(tensor, no_savanna)

    # colorize labels
    cmap = mycmap()
    imgs = []
    for s in range(tensor.shape[0]):
        im = (tensor[s, :, :] - 1) / 10
        im = cmap(im)[:, :, 0:3]
        im = np.rollaxis(im, 2, 0)
        imgs.append(im)
    tensor = np.array(imgs)

    return tensor


def classnames():
    return ["Forest", "Shrubland", "Savanna", "Grassland", "Wetlands",
            "Croplands", "Urban/Built-up", "Snow/Ice", "Barren", "Water"]


def mycmap():
    cmap = colors.ListedColormap(['#009900',
                                  '#c6b044',
                                  '#fbff13',
                                  '#b6ff05',
                                  '#27ff87',
                                  '#c24f44',
                                  '#a5a5a5',
                                  '#69fff8',
                                  '#f9ffa4',
                                  '#1c0dff',
                                  '#ffffff'])
    return cmap


def mypatches():
    patches = []
    for counter, name in enumerate(classnames()):
        patches.append(mpatches.Patch(color=mycmap().colors[counter],
                                      label=name))
    return patches


## tensor operation

def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tesnor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True


def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
    """
    assert len(clip.size()) == 4, "clip should be a 4D tensor"
    return clip[..., i:i + h, j:j + w]


def center_crop(clip, crop_size):
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
    h, w = clip.size(-2), clip.size(-1)
    th, tw = crop_size, crop_size
    assert h >= th and w >= tw, "height and width must be no smaller than crop_size"

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(clip, i, j, th, tw)


class CenterCropVideo(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, size, size)
        """
        return center_crop(clip, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


def ztz(x, y):
    """
    Compute the inner product between datapoints from corresponding patches of data
    organized in batches. Since x and y are data between the range [-1,1],
    it is normalized to be between the range [0,1] using max_norm.
        Input:x - float, array of [batch_size, patch_size, patch_size, num_channels],
                Batch of patches from data domain x.
              y - float, array of [batch_size, patch_size, patch_size, num_channels],
                Batch of patches from data domain y.
        Output:
            ztz - float, array of [batch_size, patch_size^2, patch_size^2], Inner product
    """
    max_norm = x.shape[-1]
    flat_shape = [x.shape[0], x.shape[1] ** 2, -1]
    x = torch.reshape(x, flat_shape)
    y = torch.reshape(y, flat_shape)
    #ztz = (tf.keras.backend.batch_dot(y, x, -1) + max_norm) / (2 * max_norm) ??
    ztz = (torch.bmm(x, y.permute(0, 2, 1)) + max_norm)/ (2 * max_norm)
    return ztz

def affinity(x):
    """
    Compute the affinity matrices of the patches of contained in a batch.
    It first computes the distances between the datapoints within a patch.
    Then it finds the suitable kernel width for each patch.
    Finally, applies the RBF.
        Input:
            x - float, array of [batch_size, patch_size, patch_size, num_channels],
                Batch of patches from data domain x.
        Output:
            A - float, array of [batch_size, patch_size^2, patch_size^2], Affinity matrix
    """
    _, h, w, c = x.shape
    x_1 = torch.unsqueeze(torch.reshape(x, [-1, h * w, c]), 2)
    x_2 = torch.unsqueeze(torch.reshape(x, [-1, h * w, c]), 1)
    A = torch.norm(x_1 - x_2, dim=-1)
    krnl_width, _ = torch.topk(A, k=A.shape[-1])
    krnl_width = torch.mean(krnl_width[:, :, (h * w) // 4], 1)
    krnl_width = torch.reshape(krnl_width, (-1, 1, 1))
    krnl_width = torch.where(torch.eq(krnl_width, torch.zeros_like(krnl_width)), torch.ones_like(krnl_width), krnl_width)
    A = torch.exp(-(torch.div(A, krnl_width) ** 2))
    return A

def Degree_matrix(x, y):
    """
    Compute the degree matrix starting from corresponding patches of data organized
    in batches. It first computes the affinity matrices of the two batches and then
    it computes the norm of the difference between the rows of Ax and the rows of Ay.
    Then it is normalized.
        Input:
            x - float, array of [batch_size, patch_size, patch_size, num_channels_x],
                Batch of patches from data domain x.
            y - float, array of [batch_size, patch_size, patch_size, num_channels_y],
                Batch of patches from data domain y.
        Output:
            D - float, array of [batch_size, patch_size^2, patch_size^2], Degree matrix
    """
    ax = affinity(x)
    ay = affinity(y)
    D = torch.norm(torch.unsqueeze(ax, 1) - torch.unsqueeze(ay, 2), 2, -1)
    D = (D - torch.min(D)) / (torch.max(D) - torch.min(D))
    return D

#CVA
def cva(X, Y):

    diff = X - Y
    diff_s = (diff**2).sum(axis=-1)

    return torch.sqrt(diff_s)

def SFA(X, Y):
    '''
    see http://sigma.whu.edu.cn/data/res/files/SFACode.zip
    '''
    norm_flag = True
    m, n = np.shape(X)
    meanX = np.mean(X, axis=0)
    meanY = np.mean(Y, axis=0)

    stdX = np.std(X, axis=0)
    stdY = np.std(Y, axis=0)

    Xc = (X - meanX) / stdX
    Yc = (Y - meanY) / stdY

    Xc = Xc.T
    Yc = Yc.T

    A = np.matmul((Xc-Yc), (Xc-Yc).T)/m
    B = (np.matmul(Yc, Yc.T)+np.matmul(Yc, Yc.T))/2/m

    D, V = scipy.linalg.eig(A, B)  # V is column wise
    D = D.real
    #idx = D.argsort()
    #D = D[idx]

    if norm_flag is True:
        aux1 = np.matmul(np.matmul(V.T, B), V)
        aux2 = 1/np.sqrt(np.diag(aux1))
        V = V * aux2
    #V = V[:,0:3]
    X_trans = np.matmul(V.T, Xc).T
    Y_trans = np.matmul(V.T, Yc).T

    return X_trans, Y_trans


# split whole image to patches
def patchize(img: torch.Tensor, patch_size, unfold_stride) -> torch.Tensor:
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

#thresholding methods
def kde_statsmodels_u(x, x_grid, bandwidth, **kwargs):
    kde = KDEUnivariate(x)
    kde.fit(bw=bandwidth, **kwargs)
    return kde.evaluate(x_grid)

    #Rosin
def rosin(heatmap):
    heatmap_list = heatmap.flatten().tolist()
    f_heatmap = np.array(heatmap_list)
    new_data = f_heatmap - np.min(f_heatmap)
    print(np.min(new_data))
    # declare kernel estimation parameters
    bandwidth = 0.06
    # estimate kernel
    x_grid = np.linspace(0, np.max(new_data), 90)  # x-coordinates for data points in the kernel
    kernel = kde_statsmodels_u(new_data, x_grid, bandwidth)  # get kernel

    # get the index of the kernal peak
    maxIndex = np.argmax(kernel)

    # Assign percent below the max kernel value for the 'zero' peak i.e. a value of 2 = 2% the maximum value
    maxPercent = 1

    # assign x and y coords for peak-to-base line
    x1 = x_grid[maxIndex]
    y1 = kernel[maxIndex]
    # find all local minima in the kernel
    local_mins = np.where(np.r_[True, kernel[1:] < kernel[:-1]] & np.r_[kernel[:-1] < kernel[1:], True])
    local_mins = local_mins[0]  # un 'tuple' local mins
    # filter for points below a certain kernel max
    local_mins = local_mins[(np.where(kernel[local_mins] < (y1 / (100 / maxPercent))))]
    # get local minima beyond the peak
    local_mins = local_mins[(np.where(local_mins > maxIndex))]  # get local minima that meet percent max threshold
    x2_index = local_mins[0]  # find minumum beyond peak of kernel
    x2 = x_grid[x2_index]  # index to local min beyond kernel peak
    y2 = kernel[x2_index]

    # calculate line slope and get perpendicular line
    slope = (y2 - y1) / (x2 - x1)
    # find y_intercept for line
    y_int = y1 - (slope * x1)
    slopeTan = -1 / slope  # perpendicular line slope

    # allocate lists for x-y coordinates and distance values
    dist = list()
    # save x-y coords of intersect points
    yii = list()
    xii = list()

    # iterate and generate perpendicular lines
    for i in range(maxIndex + 1, x2_index):
        # find intersection point between lines
        # determine equation of the perpendicular line based on current bin coordinate
        xt1 = x_grid[i]
        yt1 = kernel[i]
        y_int_tan = yt1 - (slopeTan * xt1)
        # calculate intersection point between lines
        b1 = y_int
        b2 = y_int_tan
        m1 = slope
        m2 = slopeTan
        # y = mx + b
        # Set both lines equal to find the intersection point in the x direction, y1=y2, x1=x2
        # y1 = m1 * x + b1, y2 = m2 * x + b2
        # if y1 == y2...
        # m1 * x + b1 = m2 * x + b2
        # m1 * x - m2 * x = b2 - b1
        # x * (m1 - m2) = b2 - b1
        # x = (b2 - b1) / (m1 - m2)
        xi = (b2 - b1) / (m1 - m2)
        # Now solve for y -- use either line, because they are equal here
        # y = mx + b
        yi = m1 * xi + b1
        # assert that the new line generated is equal or very close to the correct perpendicular value of the max deviation line
        assert ((m2 - m2 * .01) < ((yi - y_int_tan) / (xi - 0)) < (
                    m2 + m2 * .01))  # an error will throw if this statement is false
        # save x-y coordinates of the point
        yii.append(yi)
        xii.append(xi)
        # get euclidean distance between kernel coordinate and intersect point
        euc = np.sqrt((xi - xt1) ** 2 + (yi - yt1) ** 2)
        # store the euclidean distance
        dist.append(euc)

    # get kernel point with the maximum distance from the Rosin line
    # remeber, we started at maxIndex+1, so the index of the optimalPoint in the kernel array will be maxIndex+1
    # + the index in the 'dist' array
    optimalPoint = np.argmax(dist) + maxIndex + 1
    # plot the optimal point over the kernel with Rosin line we plotted before
    threshold = x_grid[optimalPoint]
    final_threhold = threshold + np.min(f_heatmap)
    #return heatmap < final_threhold
    return final_threhold


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def default(val, def_val):
    return def_val if val is None else val

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


# default SimCLR augmentation
image_size = 256
DEFAULT_AUG = nn.Sequential(
            RandomApply(augs.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
            augs.RandomGrayscale(p=0.2),
            augs.RandomHorizontalFlip(),
            RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
            augs.RandomResizedCrop((image_size, image_size)))
            #color.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])))



if __name__ == '__main__':
    meter = AverageMeter()
