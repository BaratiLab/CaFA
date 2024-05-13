import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from libs.basics import GroupNorm


class HalfPeriodicConv2d(nn.Conv2d):
    # conv2d that will pad the input in the periodic direction with circular padding
    # also will consider the latitude weight when doing the convolution
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=True,
                 periodic_direction='x', use_latitude_weight=True, polar_eps=1e-3):
        super(HalfPeriodicConv2d, self).__init__(in_channels, out_channels,
                                                  kernel_size, stride,
                                                  0, dilation,
                                                  groups, bias)
        self.periodic_direction = periodic_direction
        self.use_latitude_weight = use_latitude_weight
        self.polar_eps = polar_eps

    def pad(self, x, padding):
        # padding is a list of 4 integers, representing the padding for the 4 sides
        # padding: [left, right, top, bottom]
        # pad x according to the periodic direction
        if self.periodic_direction == 'x':
            x = F.pad(x, (padding[0], padding[1], 0, 0), mode='circular')
            x = F.pad(x, (0, 0, padding[2], padding[3]), mode='constant', value=0)
        elif self.periodic_direction == 'y':
            x = F.pad(x, (0, 0, padding[2], padding[3]), mode='circular')
            x = F.pad(x, (padding[0], padding[1], 0, 0), mode='constant', value=0)
        else:
            raise ValueError('periodic_direction must be x or y')
        return x

    def forward(self, x, padding=None, lat=None):
        # x: b c h w
        # lat: h
        # pad x according to the periodic direction
        if padding is not None:
            x = self.pad(x, padding)
        if self.use_latitude_weight:
            # add latitude weight
            lat_weight = torch.cos(lat)
            # do not die out two polar
            lat_weight[0] = self.polar_eps
            lat_weight[-1] = self.polar_eps
            x = x * lat_weight[None, None, :, None]
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        0, self.dilation, self.groups)


class PatchifyConv(nn.Module):
    # achieve patchify by conv + linear patch embedding
    def __init__(self, patch_size, in_channels, out_channels):
        super(PatchifyConv, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = HalfPeriodicConv2d(in_channels, out_channels,
                                       kernel_size=patch_size, stride=patch_size,
                                       periodic_direction='x', use_latitude_weight=False)

        # self.linear = nn.Linear(in_channels*patch_size*patch_size, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # x: b h w c
        # lat: h
        # patchify x
        # x: b h w c -> b (h//patch_size) (w//patch_size) (c)
        x = x.permute(0, 3, 1, 2)
        # determine the padding size by checking if the input size is divisible by patch_size
        h, w = x.shape[2], x.shape[3]
        padding = [0, 0, 0, 0]

        if h % self.patch_size != 0:
            padding[3] = self.patch_size - h % self.patch_size

        if w % self.patch_size != 0:
            padding[1] = self.patch_size - w % self.patch_size

        if padding != [0, 0, 0, 0]:
            x = self.conv(x, padding=padding)
        else:
            x = self.conv(x)
        # x_linear = self.linear(x_patched)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class UnpatchifyEmbd(nn.Module):
    # achieve unpatchify by conv
    def __init__(self, patch_size, in_channels, out_channels, ln=True):
        super(UnpatchifyEmbd, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        if ln:
            self.norm = nn.LayerNorm(in_channels)
        else:
            self.norm = nn.Identity()
        self.linear = nn.Linear(in_channels, out_channels*patch_size*patch_size)

    def forward(self, x):
        # lat: h
        # unpatchify x
        # x: b (h//patch_size) (w//patch_size) (c) -> b h w c
        x = self.norm(x)
        x = self.linear(x)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=self.patch_size, p2=self.patch_size)

        return x


# height un-patchify with height-wise affine and bias
class HeightUpsampleLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 in_levels,
                 out_channels,
                 out_levels):
        super(HeightUpsampleLayer, self).__init__()
        self.level_affine1 = nn.Parameter(torch.zeros(out_levels, hidden_channels), requires_grad=True)
        self.level_bias1 = nn.Parameter(torch.zeros(out_levels, hidden_channels), requires_grad=True)
        self.gn1 = GroupNorm(1, hidden_channels, affine=False)

        self.unpatchify = nn.Linear(in_channels, hidden_channels*out_levels, bias=False)
        self.to_out = nn.Linear(hidden_channels, out_channels)

        self.gn2 = GroupNorm(1, out_channels, affine=False)
        self.level_affine2 = nn.Parameter(torch.zeros(out_levels, out_channels), requires_grad=True)
        self.level_bias2 = nn.Parameter(torch.zeros(out_levels, out_channels), requires_grad=True)

        self.in_levels = in_levels
        self.out_levels = out_levels

    def forward(self, x):
        x = self.unpatchify(x)
        x = rearrange(x, 'b nlat nlon (nlev c) -> b nlat nlon nlev c', nlev=self.out_levels)
        x = F.silu(x)
        x = self.gn1(x) * (1+self.level_affine1) + self.level_bias1
        x = self.to_out(x)
        x = F.silu(x)
        x = self.gn2(x) * (1+self.level_affine2) + self.level_bias2
        return x