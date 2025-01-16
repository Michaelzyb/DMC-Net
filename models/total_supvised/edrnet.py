import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# Basic Convolution Block
class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_ch
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.01) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)

        return x


class MLP(nn.Module):
    """
    Multilayer perception block
    :param
    channels: int
        number of input/output channels
    reduction_ratio: int, default=16
        channel reduction ratio
    """
    def __init__(self, channels, reduction_ratio=16):
        super(MLP, self).__init__()
        mid_channels = channels // reduction_ratio
        self.fc1 = nn.Linear(channels, mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid_channels, channels)

    def forward(self, x):
        x = x.view(x.size(0), -1)        # flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class channelGate(nn.Module):
    def __init__(self, gate_ch, reduction_ratio=16):
        super(channelGate, self).__init__()
        self.mlp = MLP(channels=gate_ch, reduction_ratio=reduction_ratio)

    def forward(self, x):
        # global average pooling
        att1 = torch.mean(x, dim=(2, 3))
        att1 = self.mlp(att1)
        # max pooling
        att2 = torch.max(torch.max(x, dim=2)[0], dim=2)[0]
        att2 = self.mlp(att2)
        att = att1 + att2
        scale = F.sigmoid(att).unsqueeze(2).unsqueeze(3)

        return x*scale


class spatialGate(nn.Module):
    def __init__(self):
        super(spatialGate, self).__init__()
        self.spatial = BasicConv(in_ch=2, out_ch=1, kernel_size=7, stride=1, padding=3, relu=False)

    def forward(self, x):
        # max pooling
        att1 = torch.max(x, 1)[0].unsqueeze(1)
        att2 = torch.mean(x, 1).unsqueeze(1)
        att = torch.cat((att1, att2), dim=1)
        att = self.spatial(att)
        scale = F.sigmoid(att).expand_as(x)

        return x*scale


# Convolutional Block Attention Module
class cbamBlock(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(cbamBlock, self).__init__()
        self.channelGate = channelGate(gate_ch=gate_channels, reduction_ratio=reduction_ratio)   # channel attention
        self.spatialGate = spatialGate()

    def forward(self, x):
        out = self.channelGate(x)
        out = self.spatialGate(out)

        return out


# Resnet Module
def conv3x3(inplanes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = cbamBlock(planes, reduction_ratio=16)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.cbam is not None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = cbamBlock(planes*4, reduction_ratio=16)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.cbam is not None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512*block.expansion, num_classes)

        init.kaiming_normal_(self.fc.weight)     # Kaming initial
        for key in self.state_dict():
            if key.split('.')[-1] == "weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "spatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == "bias":
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=True))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)     # flatten
        x = self.fc(x)

        return x


def res_cbam(depth):

    if depth == 18:
        model = ResNet(BasicBlock, layers=[2, 2, 2, 2])

    elif depth == 34:
        model = ResNet(BasicBlock, layers=[3, 4, 6, 3])

    elif depth == 50:
        model = ResNet(Bottleneck, layers=[3, 4, 6, 3])

    elif depth == 101:
        model = ResNet(Bottleneck, layers=[3, 4, 23, 3])

    elif depth == 152:
        model = ResNet(Bottleneck, layers=[3, 8, 36, 3])

    return model


# channel shuffle
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    # shuffle
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


# Channel Weighted Block(CWB)
class CWB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CWB, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res, f = x
        x = torch.cat((res, f), 1)
        x = self.global_pool(x)
        x = self.conv1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        f = x * f
        out = f + res

        return out


# Residual Decoder block(RDB)
class RDB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RDB, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_ch)
        if in_ch != out_ch:
            self.downsample = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                                            nn.BatchNorm2d(out_ch))
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = channel_shuffle(out, groups=2)
        out = self.prelu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


# Residual Refinement Structure with 1D filters(RRS_1D)
class RRS_1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RRS_1D, self).__init__()

        # Left
        self.conv3x1_e1 = nn.Conv2d(in_ch, out_ch, (3, 1), padding=(1, 0))
        self.conv1x3_e1 = nn.Conv2d(out_ch, 64, (1, 3), padding=(0, 1))
        self.relu_e1 = nn.ReLU(inplace=True)
        self.bn_e1 = nn.BatchNorm2d(64)

        self.pool_e1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3x1_e2 = nn.Conv2d(64, 64, (3, 1), padding=(1, 0))
        self.conv1x3_e2 = nn.Conv2d(64, 64, (1, 3), padding=(0, 1))
        self.relu_e2 = nn.ReLU(inplace=True)
        self.bn_e2 = nn.BatchNorm2d(64)

        self.pool_e2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3x1_e3 = nn.Conv2d(64, 64, (3, 1), padding=(2, 0), dilation=(2, 1))
        self.conv1x3_e3 = nn.Conv2d(64, 64, (1, 3), padding=(0, 2), dilation=(1, 2))
        self.relu_e3 = nn.ReLU(inplace=True)
        self.bn_e3 = nn.BatchNorm2d(64)

        self.pool_e3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3x1_e4 = nn.Conv2d(64, 64, (3, 1), padding=(4, 0), dilation=(4, 1))
        self.conv1x3_e4 = nn.Conv2d(64, 64, (1, 3), padding=(0, 4), dilation=(1, 4))
        self.relu_e4 = nn.ReLU(inplace=True)
        self.bn_e4 = nn.BatchNorm2d(64)

        self.pool_e4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # Bridge
        self.convb = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bnb = nn.BatchNorm2d(64)
        self.relub = nn.ReLU(inplace=True)

        # Right
        self.upsample_d4 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv3x1_d4 = nn.Conv2d(64, 64, (3, 1), padding=(4, 0), dilation=(4, 1))
        self.conv1x3_d4 = nn.Conv2d(128, 64, (1, 3), padding=(0, 4), dilation=(1, 4))
        self.relu_d4 = nn.ReLU(inplace=True)
        self.bn_d4 = nn.BatchNorm2d(64)

        self.upsample_d3 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv3x1_d3 = nn.Conv2d(64, 64, (3, 1), padding=(2, 0), dilation=(2, 1))
        self.conv1x3_d3 = nn.Conv2d(128, 64, (1, 3), padding=(0, 2), dilation=(1, 2))
        self.relu_d3 = nn.ReLU(inplace=True)
        self.bn_d3 = nn.BatchNorm2d(64)

        self.upsample_d2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv3x1_d2 = nn.Conv2d(64, 64, (3, 1), padding=(1, 0))
        self.conv1x3_d2 = nn.Conv2d(128, 64, (1, 3), padding=(0, 1))
        self.relu_d2 = nn.ReLU(inplace=True)
        self.bn_d2 = nn.BatchNorm2d(64)

        self.upsample_d1 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv3x1_d1 = nn.Conv2d(64, 64, (3, 1), padding=(1, 0))
        self.conv1x3_d1 = nn.Conv2d((out_ch+64), 64, (1, 3), padding=(0, 1))
        self.relu_d1 = nn.ReLU(inplace=True)
        self.bn_d1 = nn.BatchNorm2d(64)

        self.downsample = nn.Conv2d(64, 1, kernel_size=1, bias=False)

    def forward(self, x):
        hx = x                   # identity connections

        ex1 = self.relu_e1(self.conv3x1_e1(hx))
        hx = self.relu_e1(self.bn_e1(self.conv1x3_e1(ex1)))
        hx = self.pool_e1(hx)

        ex2 = self.relu_e2(self.conv3x1_e2(hx))
        hx = self.relu_e2(self.bn_e2(self.conv1x3_e2(ex2)))
        hx = self.pool_e2(hx)

        ex3 = self.relu_e3(self.conv3x1_e3(hx))
        hx = self.relu_e3(self.bn_e3(self.conv1x3_e3(ex3)))
        hx = self.pool_e3(hx)

        ex4 = self.relu_e4(self.conv3x1_e4(hx))
        hx = self.relu_e4(self.bn_e4(self.conv1x3_e4(ex4)))
        hx = self.pool_e4(hx)

        eb = self.relub(self.bnb(self.convb(hx)))

        hx = self.upsample_d4(eb)
        hx = self.relu_d4(self.conv3x1_d4(hx))
        dx4 = self.relu_d4(self.bn_d4(self.conv1x3_d4(torch.cat((hx, ex4), 1))))

        hx = self.upsample_d3(dx4)
        hx = self.relu_d3(self.conv3x1_d3(hx))
        dx3 = self.relu_d3(self.bn_d3(self.conv1x3_d3(torch.cat((hx, ex3), 1))))

        hx = self.upsample_d2(dx3)
        hx = self.relu_d2(self.conv3x1_d2(hx))
        dx2 = self.relu_d2(self.bn_d2(self.conv1x3_d2(torch.cat((hx, ex2), 1))))

        hx = self.upsample_d1(dx2)
        hx = self.relu_d1(self.conv3x1_d1(hx))
        dx1 = self.relu_d1(self.bn_d1(self.conv1x3_d1(torch.cat((hx, ex1), 1))))

        residual = self.downsample(dx1)

        return x + residual


class EDRNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(EDRNet, self).__init__()
        resnet = res_cbam(depth=34)

        # ************************* Encoder ***************************
        # input
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Extract Features
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        # Bridge
        self.convbg_1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(512, 512, kernel_size=3, dilation=4, padding=4)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)
        # ************************* Decoder ***************************
        self.cwd0 = CWB(in_ch=128, out_ch=64)
        self.cwd1 = CWB(in_ch=128, out_ch=64)
        self.cwd2 = CWB(in_ch=256, out_ch=128)
        self.cwd3 = CWB(in_ch=512, out_ch=256)
        self.cwd4 = CWB(in_ch=1024, out_ch=512)
        # step d4
        self.up_d4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.rdb_d4_1 = RDB(in_ch=512, out_ch=512)
        self.rdb_d4_2 = RDB(in_ch=512, out_ch=256)
        # step d3
        self.up_d3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.rdb_d3_1 = RDB(in_ch=256, out_ch=256)
        self.rdb_d3_2 = RDB(in_ch=256, out_ch=128)
        # step d2
        self.up_d2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.rdb_d2_1 = RDB(in_ch=128, out_ch=128)
        self.rdb_d2_2 = RDB(in_ch=128, out_ch=64)
        # step d1
        self.up_d1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.rdb_d1_1 = RDB(in_ch=64, out_ch=64)
        self.rdb_d1_2 = RDB(in_ch=64, out_ch=64)
        # step d0
        self.up_d0 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.rdb_d0_1 = RDB(in_ch=64, out_ch=64)
        self.rdb_d0_2 = RDB(in_ch=64, out_ch=64)
        # ************************* Side Output ***************************
        self.salb = nn.Conv2d(512, num_classes, kernel_size=3, padding=1)
        self.sal4 = nn.Conv2d(256, num_classes, kernel_size=3, padding=1)
        self.sal3 = nn.Conv2d(128, num_classes, kernel_size=3, padding=1)
        self.sal2 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        self.sal1 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        self.sal0 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        # ************************* Feature Map Upsample ***************************
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upsampleb = nn.Upsample(scale_factor=32, mode='bilinear')
        # ************************* Residual Refinement Structure with 1D filters ***************************
        self.rrs_1d = RRS_1D(in_ch=num_classes, out_ch=64)

    def forward(self, x):
        # ************************* Encoder ***************************
        # input
        tx = self.conv1(x)
        tx = self.bn1(tx)
        f0 = self.relu(tx)
        tx = self.maxpool(f0)
        # Extract Features
        f1 = self.encoder1(tx)
        f2 = self.encoder2(f1)
        f3 = self.encoder3(f2)
        f4 = self.encoder4(f3)
        # Bridge
        tx = self.relubg_1(self.bnbg_1(self.convbg_1(f4)))
        tx = self.relubg_m(self.bnbg_m(self.convbg_m(tx)))
        outb = self.relubg_2(self.bnbg_2(self.convbg_2(tx)))          # 7*7*512
        # ************************* Decoder ***************************
        # step d4
        tx = self.cwd4((self.up_d4(outb), f4))
        tx = self.rdb_d4_1(tx)
        out4 = self.rdb_d4_2(tx)                                      # 14*14*256
        # step d3
        tx = self.cwd3((self.up_d3(out4), f3))
        tx = self.rdb_d3_1(tx)
        out3 = self.rdb_d3_2(tx)                                      # 28*28*128
        # step d2
        tx = self.cwd2((self.up_d2(out3), f2))
        tx = self.rdb_d2_1(tx)
        out2 = self.rdb_d2_2(tx)                                      # 56*56*64
        # step d1
        tx = self.cwd1((self.up_d1(out2), f1))
        tx = self.rdb_d1_1(tx)
        out1 = self.rdb_d1_2(tx)                                      # 112*112*64
        # step d0
        tx = self.cwd0((self.up_d0(out1), f0))
        tx = self.rdb_d0_1(tx)
        out0 = self.rdb_d0_2(tx)                                      # 224*224*64
        # ************************* Side Output ***************************
        salb = self.salb(outb)
        salb = self.upsampleb(salb)         # 7->224

        sal4 = self.sal4(out4)
        sal4 = self.upsample4(sal4)         # 14->224

        sal3 = self.sal3(out3)
        sal3 = self.upsample3(sal3)         # 28->224

        sal2 = self.sal2(out2)
        sal2 = self.upsample2(sal2)         # 56->224

        sal1 = self.sal1(out1)
        sal1 = self.upsample1(sal1)         # 112->224

        sal0 = self.sal0(out0)              # 224->224
        # ************************* Residual Refinement Structure with 1D filters ***************************
        sal_out = self.rrs_1d(sal0)

        return F.sigmoid(sal_out), F.sigmoid(sal0), F.sigmoid(sal1), F.sigmoid(sal2), F.sigmoid(sal3), F.sigmoid(sal4), F.sigmoid(salb)


