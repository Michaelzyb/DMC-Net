from typing import Tuple, Iterable
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import interpolate, relu, dropout, gelu

Tuple4i = Tuple[int, int, int, int]


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        fan_out = (m.kernel_size[0] * m.kernel_size[1] * m.out_channels) // m.groups
        nn.init.normal_(m.weight, std=(2.0 / fan_out) ** 0.5)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class MixFeedForward(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int,
                 dropout_p: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        # Depth-wise convolution
        self.conv = nn.Conv2d(hidden_features, hidden_features, (3, 3), padding=(1, 1),
                              bias=True, groups=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout_p = dropout_p

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = gelu(x)
        x = dropout(x, p=self.dropout_p, training=self.training)
        x = self.fc2(x)
        x = dropout(x, p=self.dropout_p, training=self.training)
        return x


class EfficientAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 dropout_p: float = 0.0, sr_ratio: int = 1):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f'expected dim {dim} to be a multiple of num_heads {num_heads}.')

        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout_p = dropout_p

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            sr_ratio_tuple = (sr_ratio, sr_ratio)
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio_tuple, stride=sr_ratio_tuple)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, h, w):
        q = self.q(x)
        q = rearrange(q, ('b hw (m c) -> b m hw c'), m=self.num_heads)

        if self.sr_ratio > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = self.sr(x)
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)

        x = self.kv(x)
        x = rearrange(x, 'b d (a m c) -> a b m d c', a=2, m=self.num_heads)
        k, v = x.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = rearrange(x, 'b m hw c -> b hw (m c)')
        x = self.proj(x)
        x = dropout(x, p=self.dropout_p, training=self.training)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False,
                 dropout_p: float = 0.0, drop_path_p: float = 0.0, sr_ratio: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = EfficientAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       dropout_p=dropout_p, sr_ratio=sr_ratio)
        self.drop_path_p = drop_path_p
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = MixFeedForward(dim, dim, hidden_features=dim * 4, dropout_p=dropout_p)

    def forward(self, x, h, w):
        skip = x
        x = self.norm1(x)
        x = self.attn(x, h, w)
        x = drop_path(x, p=self.drop_path_p, training=self.training)
        x = x + skip

        skip = x
        x = self.norm2(x)
        x = self.ffn(x, h, w)
        x = drop_path(x, p=self.drop_path_p, training=self.training)
        x = x + skip

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size: Tuple[int, int], stride: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x, h, w


class MixTransformerStage(nn.Module):
    def __init__(
        self,
        patch_embed: OverlapPatchEmbed,
        blocks: Iterable[TransformerBlock],
        norm: nn.LayerNorm,
    ):
        super().__init__()
        self.patch_embed = patch_embed
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm

    def forward(self, x):
        x, h, w = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, h, w)
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x


class MixTransformer(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        embed_dims: Tuple4i = (64, 128, 256, 512),
        num_heads: Tuple4i = (1, 2, 4, 8),
        qkv_bias: bool = False,
        dropout_p: float = 0.0,
        drop_path_p: float = 0.0,
        depths: Tuple4i = (3, 4, 6, 3),
        sr_ratios: Tuple4i = (8, 4, 2, 1),
    ):
        super().__init__()

        self.stages = nn.ModuleList()
        for l in range(len(depths)):
            blocks = [
                TransformerBlock(dim=embed_dims[l], num_heads=num_heads[l], qkv_bias=qkv_bias,
                                 dropout_p=dropout_p, sr_ratio=sr_ratios[l],
                                 drop_path_p=drop_path_p * (sum(depths[:l])+i) / (sum(depths)-1))
                for i in range(depths[l])
            ]
            if l == 0:
                patch_embed = OverlapPatchEmbed((7, 7), stride=4, in_chans=in_chans,
                                                embed_dim=embed_dims[l])
            else:
                patch_embed = OverlapPatchEmbed((3, 3), stride=2, in_chans=embed_dims[l - 1],
                                                embed_dim=embed_dims[l])
            norm = nn.LayerNorm(embed_dims[l], eps=1e-6)
            self.stages.append(MixTransformerStage(patch_embed, blocks, norm))

        self.init_weights()

    def init_weights(self):
        self.apply(_init_weights)

    def forward(self, x):
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


def _mit_bx(embed_dims: Tuple4i, depths: Tuple4i) -> MixTransformer:
    return MixTransformer(
        embed_dims=embed_dims,
        num_heads=(1, 2, 5, 8),
        qkv_bias=True,
        depths=depths,
        sr_ratios=(8, 4, 2, 1),
        dropout_p=0.0,
        drop_path_p=0.1,
    )


def mit_b0():
    return _mit_bx(embed_dims=(32, 64, 160, 256), depths=(2, 2, 2, 2))


def mit_b1():
    return _mit_bx(embed_dims=(64, 128, 320, 512), depths=(2, 2, 2, 2))


def mit_b2():
    return _mit_bx(embed_dims=(64, 128, 320, 512), depths=(3, 4, 6, 3))


def mit_b3():
    return _mit_bx(embed_dims=(64, 128, 320, 512), depths=(3, 4, 18, 3))


def mit_b4():
    return _mit_bx(embed_dims=(64, 128, 320, 512), depths=(3, 8, 27, 3))


def mit_b5():
    return _mit_bx(embed_dims=(64, 128, 320, 512), depths=(3, 6, 40, 3))





class SegFormerHead(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim, dropout_p=0.1, align_corners=False):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p
        self.align_corners = align_corners

        self.layers = nn.ModuleList([nn.Conv2d(chans, embed_dim, (1, 1))
                                     for chans in reversed(in_channels)])
        self.linear_fuse = nn.Conv2d(embed_dim * len(self.layers), embed_dim, (1, 1), bias=False)
        self.bn = nn.BatchNorm2d(embed_dim, eps=1e-5)
        self.rebuild_output_layer_(num_classes)

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.linear_fuse.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def rebuild_output_layer_(self, num_classes):
        self.linear_pred = nn.Conv2d(self.embed_dim, num_classes, kernel_size=(1, 1))
        self.num_classes = num_classes

    def forward(self, x):
        feats_hw = x[0].shape[2:]
        x = [layer(xi) for layer, xi in zip(self.layers, reversed(x))]
        x = [interpolate(xi, size=feats_hw, mode='bilinear', align_corners=self.align_corners)
             for xi in x[:-1]] + [x[-1]]
        x = self.linear_fuse(torch.cat(x, dim=1))
        x = self.bn(x)
        x = relu(x, inplace=True)
        x = dropout(x, p=self.dropout_p, training=self.training)
        x = self.linear_pred(x)
        return x



# This file incorporates work from https://github.com/rwightman/pytorch-image-models which is
# covered by the following copyright and permission notice:
#
#    Copyright 2019 Ross Wightman
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import math
import warnings

import torch
import torch.nn as nn


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def drop_path(x, p: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if p == 0. or not training:
        return x
    keep_prob = 1 - p
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

model_urls = {
    # Complete SegFormer weights trained on ADE20K.
    'ade': {
        'segformer_b0': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b0_512x512_ade_160k-d0c08cfd.pth',
        'segformer_b1': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b1_512x512_ade_160k-1cd52578.pth',
        'segformer_b2': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b2_512x512_ade_160k-fa162a4f.pth',
        'segformer_b3': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b3_512x512_ade_160k-5abb3eb3.pth',
        'segformer_b4': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b4_512x512_ade_160k-bb0fa50c.pth',
        'segformer_b5': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b5_640x640_ade_160k-106a5e57.pth',
    },
    # Complete SegFormer weights trained on CityScapes.
    'city': {
        'segformer_b0': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b0_1024x1024_city_160k-3e581249.pth',
        'segformer_b1': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b1_1024x1024_city_160k-e415b121.pth',
        'segformer_b2': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b2_1024x1024_city_160k-9793f658.pth',
        'segformer_b3': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b3_1024x1024_city_160k-732b9fde.pth',
        'segformer_b4': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b4_1024x1024_city_160k-1836d907.pth',
        'segformer_b5': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b5_1024x1024_city_160k-2ca4dff8.pth',
    },
    # Backbone-only SegFormer weights trained on ImageNet.
    'imagenet': {
        'segformer_b0': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b0_backbone_imagenet-eb42d485.pth',
        'segformer_b1': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b1_backbone_imagenet-357971ac.pth',
        'segformer_b2': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b2_backbone_imagenet-3c162bb8.pth',
        'segformer_b3': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b3_backbone_imagenet-0d113e32.pth',
        'segformer_b4': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b4_backbone_imagenet-b757a54d.pth',
        'segformer_b5': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b5_backbone_imagenet-d552b33d.pth',
    },
}


class SegFormer(nn.Module):
    def __init__(self, backbone: MixTransformer, decode_head: SegFormerHead):
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head

    @property
    def align_corners(self):
        return self.decode_head.align_corners

    @property
    def num_classes(self):
        return self.decode_head.num_classes

    def forward(self, x):
        image_hw = x.shape[2:]
        x = self.backbone(x)
        x = self.decode_head(x)
        x = interpolate(x, size=image_hw, mode='bilinear', align_corners=self.align_corners)
        return x


def create_segformer_b0(num_classes):
    backbone = mit_b0()
    head = SegFormerHead(
        in_channels=(32, 64, 160, 256),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=256,
    )
    return SegFormer(backbone, head)


def create_segformer_b1(num_classes):
    backbone = mit_b1()
    head = SegFormerHead(
        in_channels=(64, 128, 320, 512),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=256,
    )
    return SegFormer(backbone, head)


def create_segformer_b2(num_classes):
    backbone = mit_b2()
    head = SegFormerHead(
        in_channels=(64, 128, 320, 512),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=768,
    )
    return SegFormer(backbone, head)


def create_segformer_b3(num_classes):
    backbone = mit_b3()
    head = SegFormerHead(
        in_channels=(64, 128, 320, 512),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=768,
    )
    return SegFormer(backbone, head)


def create_segformer_b4(num_classes):
    backbone = mit_b4()
    head = SegFormerHead(
        in_channels=(64, 128, 320, 512),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=768,
    )
    return SegFormer(backbone, head)


def create_segformer_b5(num_classes):
    backbone = mit_b5()
    head = SegFormerHead(
        in_channels=(64, 128, 320, 512),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=768,
    )
    return SegFormer(backbone, head)


def _load_pretrained_weights_(model, model_url, progress):
    state_dict = torch.hub.load_state_dict_from_url(model_url, progress=progress)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('decode_head'):
            if k.endswith('.proj.weight'):
                k = k.replace('.proj.weight', '.weight')
                v = v[..., None, None]
            elif k.endswith('.proj.bias'):
                k = k.replace('.proj.bias', '.bias')
            elif '.linear_fuse.conv.' in k:
                k = k.replace('.linear_fuse.conv.', '.linear_fuse.')
            elif '.linear_fuse.bn.' in k:
                k = k.replace('.linear_fuse.bn.', '.bn.')

            if '.linear_c4.' in k:
                k = k.replace('.linear_c4.', '.layers.0.')
            elif '.linear_c3.' in k:
                k = k.replace('.linear_c3.', '.layers.1.')
            elif '.linear_c2.' in k:
                k = k.replace('.linear_c2.', '.layers.2.')
            elif '.linear_c1.' in k:
                k = k.replace('.linear_c1.', '.layers.3.')
        else:
            if 'patch_embed1.' in k:
                k = k.replace('patch_embed1.', 'stages.0.patch_embed.')
            elif 'patch_embed2.' in k:
                k = k.replace('patch_embed2.', 'stages.1.patch_embed.')
            elif 'patch_embed3.' in k:
                k = k.replace('patch_embed3.', 'stages.2.patch_embed.')
            elif 'patch_embed4.' in k:
                k = k.replace('patch_embed4.', 'stages.3.patch_embed.')
            elif 'block1.' in k:
                k = k.replace('block1.', 'stages.0.blocks.')
            elif 'block2.' in k:
                k = k.replace('block2.', 'stages.1.blocks.')
            elif 'block3.' in k:
                k = k.replace('block3.', 'stages.2.blocks.')
            elif 'block4.' in k:
                k = k.replace('block4.', 'stages.3.blocks.')
            elif 'norm1.' in k:
                k = k.replace('norm1.', 'stages.0.norm.')
            elif 'norm2.' in k:
                k = k.replace('norm2.', 'stages.1.norm.')
            elif 'norm3.' in k:
                k = k.replace('norm3.', 'stages.2.norm.')
            elif 'norm4.' in k:
                k = k.replace('norm4.', 'stages.3.norm.')

            if '.mlp.dwconv.dwconv.' in k:
                k = k.replace('.mlp.dwconv.dwconv.', '.mlp.conv.')

            if '.mlp.' in k:
                k = k.replace('.mlp.', '.ffn.')
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)


def segformer_b0_ade(pretrained=True, progress=True):
    """Create a SegFormer-B0 model for the ADE20K segmentation task.
    """
    model = create_segformer_b0(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b0'], progress=progress)
    return model


def segformer_b1_ade(pretrained=True, progress=True):
    """Create a SegFormer-B1 model for the ADE20K segmentation task.
    """
    model = create_segformer_b1(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b1'], progress=progress)
    return model


def segformer_b2_ade(pretrained=True, progress=True):
    """Create a SegFormer-B2 model for the ADE20K segmentation task.
    """
    model = create_segformer_b2(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b2'], progress=progress)
    return model


def segformer_b3_ade(pretrained=True, progress=True):
    """Create a SegFormer-B3 model for the ADE20K segmentation task.
    """
    model = create_segformer_b3(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b3'], progress=progress)
    return model


def segformer_b4_ade(pretrained=True, progress=True):
    """Create a SegFormer-B4 model for the ADE20K segmentation task.
    """
    model = create_segformer_b4(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b4'], progress=progress)
    return model


def segformer_b5_ade(pretrained=True, progress=True):
    """Create a SegFormer-B5 model for the ADE20K segmentation task.
    """
    model = create_segformer_b5(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b5'], progress=progress)
    return model


def segformer_b0_city(pretrained=True, progress=True):
    """Create a SegFormer-B0 model for the CityScapes segmentation task.
    """
    model = create_segformer_b0(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b0'], progress=progress)
    return model


def segformer_b1_city(pretrained=True, progress=True):
    """Create a SegFormer-B1 model for the CityScapes segmentation task.
    """
    model = create_segformer_b1(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b1'], progress=progress)
    return model


def segformer_b2_city(pretrained=True, progress=True):
    """Create a SegFormer-B2 model for the CityScapes segmentation task.
    """
    model = create_segformer_b2(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b2'], progress=progress)
    return model


def segformer_b3_city(pretrained=True, progress=True):
    """Create a SegFormer-B3 model for the CityScapes segmentation task.
    """
    model = create_segformer_b3(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b3'], progress=progress)
    return model


def segformer_b4_city(pretrained=True, progress=True):
    """Create a SegFormer-B4 model for the CityScapes segmentation task.
    """
    model = create_segformer_b4(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b4'], progress=progress)
    return model


def segformer_b5_city(pretrained=True, progress=True):
    """Create a SegFormer-B5 model for the CityScapes segmentation task.
    """
    model = create_segformer_b5(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b5'], progress=progress)
    return model


def segformer_b0(pretrained=False, progress=True, num_classes=150):
    """Create a SegFormer-B0 model.

    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b0(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b0'],
                                  progress=progress)
    return model


def segformer_b1(pretrained=False, progress=True, num_classes=150):
    """Create a SegFormer-B1 model.

    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b1(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b1'],
                                  progress=progress)
    return model


def segformer_b2(pretrained=False, progress=True, num_classes=150):
    """Create a SegFormer-B2 model.

    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b2(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b2'],
                                  progress=progress)
    return model


def segformer_b3(pretrained=False, progress=True, num_classes=150):
    """Create a SegFormer-B3 model.

    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b3(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b3'],
                                  progress=progress)
    return model


def segformer_b4(pretrained=False, progress=True, num_classes=150):
    """Create a SegFormer-B4 model.

    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b4(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b4'],
                                  progress=progress)
    return model


def segformer_b5(pretrained=False, progress=True, num_classes=150):
    """Create a SegFormer-B5 model.

    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b5(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b5'],
                                  progress=progress)
    return model
