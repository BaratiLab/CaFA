import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange, repeat


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))


class GeAct(nn.Module):
    """Gated activation function"""
    def __init__(self, act_fn):
        super().__init__()
        self.fn = act_fn

    def forward(self, x):
        c = x.shape[-1]  # channel last arrangement
        return self.fn(x[..., :int(c//2)]) * x[..., int(c//2):]


class MLP(nn.Module):
    def __init__(self, dims, act_fn, dropout=0., no_bias=True):
        super().__init__()
        layers = []

        for i in range(len(dims) - 1):
            if isinstance(act_fn, GeAct) and i < len(dims) - 2:
                layers.append(nn.Linear(dims[i], dims[i+1] * 2, bias=not no_bias))
            else:
                layers.append(nn.Linear(dims[i], dims[i+1], bias=not no_bias))
            if i < len(dims) - 2:
                layers.append(act_fn)
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GroupNorm(nn.Module):
    # group norm with channel at the last dimension
    def __init__(self, num_groups, num_channels,
                 domain_wise=False,
                 eps=1e-8, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.domain_wise = domain_wise
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(num_channels), requires_grad=True)

    def forward(self, x):
        # b h w c
        b, c = x.shape[0], x.shape[-1]
        if self.domain_wise:
            x = rearrange(x, 'b ... (g c) -> b g (... c)', g=self.num_groups)
        else:
            x = rearrange(x, 'b ... (g c) -> b ... g c', g=self.num_groups)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        if self.domain_wise:
            x = rearrange(x, 'b g (... c) -> b ... (g c)',
                          g=self.num_groups)
        else:
            x = rearrange(x, 'b ... g c -> b ... (g c)',
                          g=self.num_groups)
        if self.affine:
            x = x * self.weight + self.bias
        return x


class InstanceNorm(nn.Module):
    # instance norm with channel at the last dimension
    def __init__(self, num_channels, eps=1e-6, affine=False):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(num_channels), requires_grad=True)

    def forward(self, x):
        # b h w c
        shape = x.shape
        # collapse all spatial dimension
        x = x.reshape(shape[0], -1, shape[-1])
        mean = x.mean(dim=-2, keepdim=True)
        var = x.var(dim=-2, keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        if self.affine:
            x = x * self.weight + self.bias
        # restore the spatial dimension
        x = x.reshape(shape)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8, affine=True):
        super().__init__()
        self.eps = eps
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim), requires_grad=affine)

    def forward(self, x):
        return F.normalize(x, dim=-1, eps=self.eps) * self.g * self.scale
