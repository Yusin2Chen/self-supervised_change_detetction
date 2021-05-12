import torch
from torch.nn import init
import torch.nn as nn
import math
import numpy as np
from torch.nn import functional as F
from utils.util import center_crop, ztz, SFA


class CouplingNetwork(nn.Module):
    def __init__(self, in_chs, filter, decoder=False, l2_lamada=1e-3):
        super(CouplingNetwork, self).__init__()
        self.filter = filter
        self.decoder = decoder
        self.layer0 = nn.Sequential(nn.ReplicationPad2d(1),
                                    nn.Conv2d(in_chs, self.filter[0], kernel_size=3, stride=1))
        self.layer1 = self._make_layer(nn.Conv2d)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, kernel_size=1, stride=1, padding=0, dilate=False):
        layers = []
        for i in range(1, len(self.filter)):
            layers.append(nn.Sigmoid())
            layers.append(block(self.filter[i-1], self.filter[i], kernel_size, stride, padding))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        if self.decoder:
            x = self.tanh(x)
        else:
            x = self.sigm(x)
        return x


class SCCN(nn.Module):
    def __init__(self, in_enc, ot_dec, kernel_size=1, stride=1):
        super(SCCN, self).__init__()
        self.l2_lambda = 1e-6
        self.Lambda = 1
        self.enc_filter = [20, 20, 20, 20]
        self.dec_filter = [ot_dec]
        self._enc_x = CouplingNetwork(in_enc, self.enc_filter)
        self._enc_y = CouplingNetwork(in_enc, self.enc_filter)
        self._dec_x = CouplingNetwork(self.enc_filter[-1], self.dec_filter, decoder=True)
        self._dec_y = CouplingNetwork(self.enc_filter[-1], self.dec_filter, decoder=True)

    def _domain_difference_img(self, original, transformed, bandwidth=3):
        """
            Compute difference image in one domain between original image
            in that domain and the transformed image from the other domain.
            Bandwidth governs the norm difference clipping threshold
        """
        d = torch.norm(original - transformed, p=2, dim=1)
        threshold = torch.mean(d) + bandwidth * torch.std(d)
        d = torch.where(d < threshold, d, threshold)
        return d / torch.max(d)

    def forward(self, x, y, training=False, pretraining=False):
        if training:
            x_code, y_code = self._enc_x(x), self._enc_y(y)
            if pretraining:
                x_tilde, y_tilde = self._dec_x(x_code), self._dec_y(y_code)
                return x_tilde, y_tilde
            else:
                return x_code, y_code

        else:
            x_code, y_code = self._enc_x(x), self._enc_y(y)
            difference_img = self._domain_difference_img(x_code, y_code)
            return difference_img

class ImgTransNet(nn.Module):
    def __init__(self, in_chs, filter, l2_lamada=1e-3, leaky_alpha=0.3, dropout_rate=0.2):
        super(ImgTransNet, self).__init__()
        self.filter = filter
        self.leaky_alpha = leaky_alpha
        self.layer0 = nn.Sequential(nn.ReplicationPad2d(1),
                                    nn.Conv2d(in_chs, self.filter[0], kernel_size=3, stride=1))
        self.dropout = nn.Dropout2d(dropout_rate)
        self.layer1 = self._make_layer(nn.Conv2d)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, kernel_size=3, stride=1, padding=0, dilate=False):
        layers = []
        for i in range(1, len(self.filter)):
            layers.append(nn.LeakyReLU(negative_slope=self.leaky_alpha, inplace=True))
            layers.append(self.dropout)
            layers.append(nn.ReplicationPad2d(1))
            layers.append(block(self.filter[i - 1], self.filter[i], kernel_size, stride, padding))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.tanh(x)
        return x


class CAA(nn.Module):
    def __init__(self, in_enc, me_mec, kernel_size=1, stride=1):
        super(CAA, self).__init__()
        self.cycle_lambda = 0.2
        self.cross_lambda = 0.1
        self.recon_lambda = 0.1
        self.l2_lambda = 1e-6
        self.kernels_lambda = 1
        self.min_impr = 1e-2
        self.last_losses = []
        self.patience = 11
        self.aps = 20
        self.enc_filter = [50, 50, me_mec]
        self.dec_filter = [50, 50, in_enc]

        # encoders of X and Y
        self._enc_x = ImgTransNet(in_enc, self.enc_filter)
        self._enc_y = ImgTransNet(in_enc, self.enc_filter)
        # decoder of X and Y
        self._dec_x = ImgTransNet(me_mec, self.dec_filter)
        self._dec_y = ImgTransNet(me_mec, self.dec_filter)

    def _domain_difference_img(self, original, transformed, bandwidth=3):
        """
            Compute difference image in one domain between original image
            in that domain and the transformed image from the other domain.
            Bandwidth governs the norm difference clipping threshold
        """
        d = torch.norm(original - transformed, p=2, dim=1)
        threshold = torch.mean(d) + bandwidth * torch.std(d)
        d = torch.where(d < threshold, d, threshold)
        return d / torch.max(d)

    def _difference_img(self, x, y, x_hat, y_hat):
        """
        Should compute the two possible change maps and do the 5 method
        ensamble to output a final change-map?
        """
        assert x.shape[0] == y.shape[0] == 1, "Can not handle batch size > 1"

        d_x = self._domain_difference_img(x, x_hat)
        d_y = self._domain_difference_img(y, y_hat)

        # Weighted average based on the number of estimated channels
        c_x, c_y = x.shape[-1], y.shape[-1]
        d = (c_y * d_x + c_x * d_y) / (c_x + c_y)

        # Return expanded dims (rank = 4)?
        return d

    def forward(self, x, y, training=False):
        if training:
            x_code, y_code = self._enc_x(x), self._enc_y(y)
            x_hat, y_hat = self._dec_x(y_code), self._dec_y(x_code)
            x_dot = self._dec_x(self._enc_y(y_hat))
            y_dot = self._dec_y(self._enc_x(x_hat))
            x_tilde, y_tilde = self._dec_x(x_code), self._dec_y(y_code)
            crop_x_code = center_crop(x_code, int(0.2 * x_code.shape[-1]))
            crop_y_code = center_crop(y_code, int(0.2 * y_code.shape[-1]))
            zx_t_zy = ztz(crop_x_code.permute(0, 2, 3, 1).contiguous(), crop_y_code.permute(0, 2, 3, 1).contiguous())
            return x_hat, y_hat, x_dot, y_dot, x_tilde, y_tilde, zx_t_zy
        else:
            x_code, y_code = self._enc_x(x), self._enc_y(y)
            x_tilde, y_tilde = self._dec_x(x_code), self._dec_y(y_code)
            x_hat, y_hat = self._dec_x(y_code), self._dec_y(x_code)
            difference_img = self._difference_img(x_tilde, y_tilde, x_hat, y_hat)
            return difference_img


class DSFANet(nn.Module):
    def __init__(self, num):
        super(DSFANet, self).__init__()
        self.num = num
        self.filter = [4, 128, 128]
        self.output_num = 6
        self.layers =2
        self.reg = 1e-4
        self.activation = nn.Softsign()
        self.dropout = nn.Dropout(0.2)
        self.layer0 = self._make_layer(nn.Linear)
        self.layer1 = nn.Linear(self.filter[-1], self.output_num, bias=True)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layer(self, block):
        layers = []
        for i in range(len(self.filter)-1):
            layers.append(block(self.filter[i], self.filter[i+1], bias=True))
            #layers.append(self.dropout)
            layers.append(self.activation)
        return nn.Sequential(*layers)

    def DSFA(self, X, Y):

        #m,n = troech.shape(X)
        X_hat = X - torch.mean(X, dim=0)
        Y_hat = Y - torch.mean(Y, dim=0)
        differ = X_hat - Y_hat
        A = torch.matmul(differ.T, differ)
        A = A / self.num

        Sigma_XX = torch.matmul(X_hat.T, X_hat)
        Sigma_XX = Sigma_XX / self.num + self.reg * torch.eye(self.output_num)
        Sigma_YY = torch.matmul(Y_hat.T, Y_hat)
        Sigma_YY = Sigma_YY / self.num + self.reg * torch.eye(self.output_num)
        B = (Sigma_XX + Sigma_YY) / 2

        # For numerical stability.
        D_B, V_B = torch.symeig(B, eigenvectors=True, upper=False) # 只有下三角矩阵参与运算
        #print(D_B.shape, V_B.shape)
        # 取回满足条件索引
        idx = (D_B > 1e-12).nonzero(as_tuple=False)
        idx = idx[:, 0]
        # 实现tf.gather
        D_B = torch.gather(D_B, 0, idx)
        idx_vb = idx.unsqueeze(1)
        idx_vb = idx_vb.expand(1, idx_vb.shape[0], V_B.shape[1])
        idx_vb = idx_vb.squeeze()
        V_B = torch.gather(V_B, 1, idx_vb)
        B_inv = torch.matmul(torch.matmul(V_B, torch.diag(torch.reciprocal(D_B))), V_B.T)

        Sigma = torch.matmul(B_inv, A)
        loss = torch.trace(torch.matmul(Sigma, Sigma))

        return loss

    def forward(self, X, Y, training=False, inference=False):
        #B, C, W, H = X.shape
        loss_all = []
        X = self.layer0(X)
        Y = self.layer0(Y)
        X = self.layer1(X)
        Y = self.layer1(Y)
        if training:
            #for i in range(X.shape[0]):
            #    loss = self.DSFA(X[i, :, :].squeeze(), Y[i, :, :].squeeze())
            #    loss_all.append(loss)
            #return torch.mean(torch.stack(loss_all), 0)
            loss = self.DSFA(X.squeeze().cpu(), Y.squeeze().cpu())
            return loss
        #elif inference:
        #    X_trans, Y_trans = SFA(X.squeeze().cpu().detach().numpy(), Y.squeeze().cpu().detach().numpy())
        #    diff = X_trans - Y_trans
        #    diff = diff / np.std(diff, axis=0)
        #    diff = (diff ** 2).sum(axis=1).reshape((W, H))
        #    return diff
        else:
            return X, Y
