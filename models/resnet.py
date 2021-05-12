import torch
import torch.nn as nn
import math

__all__ = ['ResNet', ]

class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

def Net6(**kwargs):
    return NetN(BasicBlock, [2, 2, 2, 2], **kwargs)

def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_feats=128, width=1, in_channel=3, mi=False):
        super(ResNet, self).__init__()

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = max(int(64 * width), 64)
        self.base = int(64 * width)
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channel, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False),
            self._norm_layer(self.inplanes),
            nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=1)
        self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.base * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if mi:
            self.final = nn.Linear(512 * block.expansion, num_feats)
        else:
            self.final = MLPHead(512 * block.expansion, 256, num_feats)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer0(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.final(x)

        return x


class NetN(nn.Module):

    def __init__(self, block, layers, num_feats=64, width=1, in_channel=3, mi=False):
        super(NetN, self).__init__()

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = max(int(64 * width), 64)
        self.base = int(64 * width)
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            self._norm_layer(self.inplanes),
            nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.base, layers[0], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if mi:
            self.final = nn.Linear(self.base, num_feats)
        else:
            self.final = MLPHead(self.base, self.base * 2, num_feats)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer0(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.final(x)

        return x