import torch
import torch.nn as nn
from torch.nn.functional import interpolate

channel_config = [32, 64, 128, 256, 512, 512]


class PGANet(nn.Module):
    def __init__(self, in_c, num_classes):
        super(PGANet, self).__init__()
        self.stage = nn.Sequential(nn.Conv2d(in_c, channel_config[0], kernel_size=3, padding=1), nn.MaxPool2d(2, 2))
        self.e1 = EncoderBlock(channel_config[0], channel_config[1], 2)
        self.e2 = EncoderBlock(channel_config[1], channel_config[2], 2)
        self.e3 = EncoderBlock(channel_config[2], channel_config[3], 3)
        self.e4 = EncoderBlock(channel_config[3], channel_config[4], 3)
        self.e5 = EncoderBlock(channel_config[4], channel_config[5], 3, is_maxpool=False)

        self.pff1 = PFFModule(1)
        self.pff2 = PFFModule(2)
        self.pff3 = PFFModule(3)
        self.pff4 = PFFModule(4)
        self.pff5 = PFFModule(4)

        self.gca2 = GCAModule(2)
        self.gca3 = GCAModule(3)
        self.gca4 = GCAModule(4)
        self.gca5 = GCAModule(4)

        self.b1 = BRModule(128)
        self.b2 = BRModule(128)
        self.b3 = BRModule(128)
        self.b4 = BRModule(128)
        self.b5 = BRModule(128)

        self.c1 = nn.Conv2d(128, num_classes, 1)
        self.c2 = nn.Conv2d(128, num_classes, 1)
        self.c3 = nn.Conv2d(128, num_classes, 1)
        self.c4 = nn.Conv2d(128, num_classes, 1)
        self.c5 = nn.Conv2d(128, num_classes, 1)
        self.final = nn.Conv2d(num_classes*5, num_classes, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.stage(x)
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)

        d1 = self.pff1([e1, e2, e3, e4, e5])
        d2 = self.pff2([e1, e2, e3, e4, e5])
        d3 = self.pff3([e1, e2, e3, e4, e5])
        d4 = self.pff4([e1, e2, e3, e4, e5])
        d5 = self.pff5([e1, e2, e3, e4, e5])

        f1 = self.b1(d1)
        f2 = self.b2(self.gca2(f1, d2))
        f3 = self.b3(self.gca3(f2, d3))
        f4 = self.b4(self.gca4(f3, d4))
        f5 = self.b5(self.gca5(f4, d5))

        f1 = interpolate(self.c1(f1), (h, w), mode='bilinear')
        f2 = interpolate(self.c2(f2), (h, w), mode='bilinear')
        f3 = interpolate(self.c3(f3), (h, w), mode='bilinear')
        f4 = interpolate(self.c4(f4), (h, w), mode='bilinear')
        f5 = interpolate(self.c5(f5), (h, w), mode='bilinear')
        final = self.final(torch.cat((f1, f2, f3, f4, f5), dim=1))
        return final, f1, f2, f3, f4, f5


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks, is_maxpool=True):
        super(EncoderBlock, self).__init__()
        layer = []
        if is_maxpool:
            layer.append(nn.MaxPool2d(2, 2))
        layer.append(nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        ))
        for i in range(1, num_blocks):
            layer.append(nn.Sequential(
                nn.Conv2d(out_c, out_c, kernel_size=3, bias=False, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            ))
        self.layer = nn.Sequential(*list(layer))

    def forward(self, x):
        x = self.layer(x)
        return x


class PFFModule(nn.Module):
    def __init__(self, index):
        super(PFFModule, self).__init__()
        self.layer = nn.ModuleList()
        for i in range(1, 6):
            if i == 5:
                i = i - 1
            k = 2 ** (index - i)
            p = 2 ** (i - index)
            if i < index:
                self.layer.append(nn.Conv2d(channel_config[i], 128, kernel_size=k, stride=k))
            elif i == index:
                self.layer.append(nn.Conv2d(channel_config[i], 128, kernel_size=1))
            else:
                self.layer.append(nn.ConvTranspose2d(channel_config[i], 128, kernel_size=p, stride=p))
        self.conv = nn.Sequential(nn.Conv2d(640, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128), nn.ReLU())

    def forward(self, features):
        for i in range(5):
            features[i] = self.layer[i](features[i])
        features = torch.cat(features, dim=1)
        features = self.conv(features)
        return features


class GCAModule(nn.Module):
    def __init__(self, index):
        super(GCAModule, self).__init__()
        self.conv_h = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv_l = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        p = 2 ** (index - 1)
        self.dconv = nn.ConvTranspose2d(128, 128, kernel_size=p, stride=p)

    def forward(self, high_f, low_f):
        high_f = self.conv_h(high_f)
        low_f = self.conv_l(low_f)
        attention = torch.mean(low_f, dim=(2, 3), keepdim=True)
        high_f = attention * high_f
        low_f = self.dconv(low_f)
        return low_f + high_f


class BRModule(nn.Module):
    def __init__(self, in_c):
        super(BRModule, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=3, padding=1), nn.ReLU())
        self.conv2 = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1)

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        return res + x


if __name__ == '__main__':
    model = PGANet(3, 4).cuda()
    x = torch.rand(1, 3, 512, 256).cuda()
    result = model(x)
    print(result)