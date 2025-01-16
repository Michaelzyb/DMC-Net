import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers, Block=BasicBlock, attention=None):
        super(_DecoderBlock, self).__init__()
        middle_channels = in_channels // 2
        layers = [
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            Block(in_channels, middle_channels)
        ]
        if attention:
            layers += [attention(middle_channels)]
        layers += [
                    Block(middle_channels, middle_channels)
                  ] * (num_conv_layers - 2)
        layers += [
            Block(middle_channels, out_channels)
        ]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class SegNet(nn.Module):
    def __init__(self, in_channel, num_classes, Block=BasicBlock, Attention=None):
        super(SegNet, self).__init__()
        vgg = models.vgg19_bn()
        vgg.features[0] = nn.Conv2d(in_channel, 64, kernel_size=3, padding=1)
        # if pretrained:
        #     vgg.load_state_dict(torch.load(vgg19_bn_path))
        features = list(vgg.features.children())
        self.enc1 = nn.Sequential(*features[0:7])
        self.enc2 = nn.Sequential(*features[7:14])
        self.enc3 = nn.Sequential(*features[14:27])
        self.enc4 = nn.Sequential(*features[27:40])
        self.enc5 = nn.Sequential(*features[40:])

        self.dec5 = nn.Sequential(
            *([nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)] + [Block(512, 512)] * 4)
        )
        self.dec4 = _DecoderBlock(1024, 256, 4, Block, Attention)
        self.dec3 = _DecoderBlock(512, 128, 4, Block, Attention)
        self.dec2 = _DecoderBlock(256, 64, 2, Block, Attention)
        self.dec1 = _DecoderBlock(128, num_classes, 2, Block, Attention)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec5 = self.dec5(enc5)
        dec4 = self.dec4(torch.cat([dec5, F.interpolate(enc4, dec5.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear')], 1))
        return dec1
