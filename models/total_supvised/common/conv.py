import math
import torch
from torch import nn
from .attention import SE


# 标准1x1卷积
class PosConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(self.bn(x))
        return x


# 标准3x3卷积
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# ResNet18的基本模块+通道注意力，效果很好
class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_c, out_c, groups=1):
        super().__init__()
        self.c1 = nn.Conv2d(in_c, out_c, 3, 1, 1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(True)
        self.c2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        if in_c == out_c:
            self.res = nn.Identity()
        else:
            self.res = nn.Conv2d(in_c, out_c, 1)
        if out_c > 16:
            reduction = 4
        else:
            reduction = 2
        self.att = SE(out_c, reduction=reduction)

    def forward(self, x):
        res = self.res(x)
        x = self.c1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(x)
        x = self.relu(res+x)
        x = self.att(x)
        return x


class MRAttentionBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.c1 = nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False)
        self.c1_hor = nn.Conv2d(in_c, out_c, (1, 3), 1, (0, 1), bias=False)
        self.c1_ver = nn.Conv2d(in_c, out_c, (3, 1), 1, (1, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.bn1_hor = nn.BatchNorm2d(out_c)
        self.bn1_ver = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(True)
        self.c2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.c2_hor = nn.Conv2d(out_c, out_c, (1, 3), 1, (0, 1), bias=False)
        self.c2_ver = nn.Conv2d(out_c, out_c, (3, 1), 1, (1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.bn2_hor = nn.BatchNorm2d(out_c)
        self.bn2_ver = nn.BatchNorm2d(out_c)
        if in_c == out_c:
            self.res = nn.Identity()
        else:
            self.res = nn.Conv2d(in_c, out_c, 1)
        if out_c > 16:
            reduction = 4
        else:
            reduction = 2
        self.att = SE(out_c, reduction=reduction)

    def forward(self, x):
        res = self.res(x)
        x = self.bn1(self.c1(x)) + self.bn1_hor(self.c1_hor(x)) + self.bn1_ver(self.c1_ver(x))
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(self.c2(x)) + self.bn2_hor(self.c2_hor(x)) + self.bn2_ver(self.c2_ver(x))
        x = self.relu(res+x)
        x = self.att(x)
        return x

# MobileNetV3的倒残差结构
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4):
        super().__init__()
        mid_channels = in_channels * expansion
        self.posconv1 = PosConv(in_channels, mid_channels)
        self.midconv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.att = SE(mid_channels, reduction=4)
        self.posconv2 = PosConv(mid_channels, out_channels)
        if in_channels != out_channels:
            self.res = False
        else:
            self.res = True

    def forward(self, x):
        out = self.posconv1(x)
        out = self.midconv(out)
        out = self.att(out)
        out = self.posconv2(out)
        if self.res:
            out = x + out
        return out


# Res2Net里的基本模块
class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, 1, bias=False), nn.BatchNorm2d(planes))

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        # 对于stype=='stage‘的时候不用加上前一小块的输出结果，而是直接 sp = spx[i]
        # 是因为输入输出的尺寸不一致（通道数不一样），所以没法加起来
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        # 在这里需要加pool的原因是因为，对于每一个layer的stage模块，它的stride是不确定，layer1的stride=1
        # layer2、3、4的stride=2，前三小块都经过了stride=2的3*3卷积，而第四小块是直接送到y中的，但它必须要pool一下
        # 不然尺寸和不能和前面三个小块对应上，无法完成最后的econcat操作
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out