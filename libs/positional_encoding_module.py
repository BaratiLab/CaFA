import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
import numpy as np


class FourierEmbedding(nn.Module):
    def __init__(self, dim, heads,
                 scale=1000,
                 trainable=True,
                 enforce_periodicity=False):
        super().__init__()
        freq_bands = 1. / (scale ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('freq_bands', freq_bands)
        self.weights = nn.Parameter(torch.randn(heads, dim) / np.sqrt(dim), requires_grad=trainable)
        self.enforce_periodicity = enforce_periodicity

    def forward(self, angle):
        # angles [n x n] assuming in radians like [0, pi]

        if self.enforce_periodicity:
            # theta = min(theta, 2pi - theta)
            angle = torch.min(angle, 2 * np.pi - angle)

        freqs = torch.einsum('i j, c -> i j c', angle, self.freq_bands)
        basis = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        return torch.einsum('i j c, h c -> h i j', basis, self.weights)


class ExponentialModulation(nn.Module):
    def __init__(
            self,
            num_kernels,
            num_heads,
            fast_decay_pct=3,
            slow_decay_pct=25,
            target=1e-2,
            enforce_periodicity=False,
            trainable=False,
    ):
        super().__init__()
        max_decay = np.log(target) / fast_decay_pct
        min_decay = np.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, num_kernels)
        self.deltas = nn.Parameter(deltas, requires_grad=trainable)
        self.weights = nn.Parameter(torch.randn(num_heads, num_kernels)*0.02, requires_grad=True)
        self.bias = nn.Parameter(torch.ones(num_heads), requires_grad=True)

        self.enforce_periodicity = enforce_periodicity

    def forward(self, angle):
        # angles [n x n] assuming in radians like [0, pi]

        if self.enforce_periodicity:
            # theta = min(theta, 2pi - theta)
            angle = torch.min(angle, 2 * np.pi - angle)
        deltas = self.deltas.abs()
        decay = torch.exp(-torch.einsum('i j, d -> i j d', angle, deltas))
        decay = torch.einsum('i j d, h d -> h i j', decay, self.weights) + self.bias.unsqueeze(-1).unsqueeze(-1)
        return decay


class RadialBesselBasis(nn.Module):
    def __init__(
            self,
            num_kernels,
            num_heads,
            dim_head,
            enforce_periodicity=False,
            trainable=False,
            act_fn=None,
    ):
        super().__init__()
        freqs = torch.arange(1, num_kernels+1).float()
        self.freqs = nn.Parameter(freqs, requires_grad=trainable)
        self.weights = nn.Parameter(torch.randn(num_heads*dim_head, num_kernels) / np.sqrt(dim_head), requires_grad=True)
        self.bias = nn.Parameter(torch.ones(num_heads*dim_head) / dim_head, requires_grad=True)

        self.enforce_periodicity = enforce_periodicity
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.act_fn = act_fn

    def forward(self, angle, cache=True):
        # angles [n x n] assuming in radians like [0, pi]
        if not cache or not hasattr(self, 'angle'):
            if self.enforce_periodicity:
                # theta = min(theta, 2pi - theta)
                angle = torch.min(angle, 2 * np.pi - angle)
            # add a small epsilon to the zero element in angle to avoid division by zero
            angle[angle == 0] = 1e-5
            if cache:
                self.angle = angle.detach()
        else:
            angle = self.angle
        theta = torch.einsum('i j, d -> i j d', angle, self.freqs)

        basis = torch.sin(theta) / theta * np.sqrt(2 / np.pi)
        decay = torch.einsum('i j c, d c -> i j d', basis, self.weights)\
                + self.bias.unsqueeze(0).unsqueeze(0)
        decay = rearrange(decay, 'i j (h d) -> h i j d', h=self.num_heads)
        return self.act_fn(decay) if self.act_fn is not None else decay


# https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


def get_time_embedding(t, dim):
    assert len(t.shape) == 1

    half_dim = dim // 2
    emb = torch.arange(half_dim, dtype=torch.float32) * 2*np.pi
    emb = emb.to(device=t.device)
    emb = t.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


# Gaussian Fourier features
# code modified from: https://github.com/ndahlquist/pytorch-fourier-feature-networks
# author: Nic Dahlquist
class GaussianFourierFeatureTransform(nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, n, num_input_channels],
     returns a tensor of size [batches, n, mapping_size*2].
    """

    def __init__(self, num_input_channels,
                 mapping_size=256, scale=10, learnable=False,
                 num_heads=1):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size

        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size * num_heads)) * scale,
                               requires_grad=learnable)
        self.num_heads = num_heads

    def forward(self, x, unfold_head=False):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        batches, num_of_points, channels = x.shape

        # Make shape compatible for matmul with _B.
        # From [B, N, C] to [(B*N), C].
        x = rearrange(x, 'b n c -> (b n) c')

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = rearrange(x, '(b n) c -> b n c', b=batches)

        x = 2 * np.pi * x
        if unfold_head:
            x = rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


# helpers

def exists(val):
    return val is not None


def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


# sin activation
class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# siren layer
class Siren(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 w0=1.,
                 c=6.,
                 is_first=False,
                 use_bias=True,
                 activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (np.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


# siren network
class SirenNet(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_hidden, dim_out, num_layers,
                 w0=1.,
                 w0_initial=30.,
                 use_bias=True, final_activation=None,
                 normalize_input=True):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.normalize_input = normalize_input

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first,
            ))

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in=dim_hidden,
                                dim_out=dim_out,
                                w0=w0,
                                use_bias=use_bias,
                                activation=final_activation)

        # self.last_layer = nn.Linear(dim_hidden, dim_out)
        # init last layer orthogonally
        # nn.init.orthogonal_(self.last_layer.weight, gain=1/dim_out)

    def forward(self, x, mods=None):
        # x = (x - 0.5) * 2

        for layer in self.layers:
            x = layer(x)
        if mods is not None:
            x *= mods
        x = self.last_layer(x)
        # x = self.final_activation(x)
        return x



