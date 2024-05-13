import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_, orthogonal_
from .positional_encoding_module import RotaryEmbedding, GaussianFourierFeatureTransform, \
    apply_rotary_pos_emb, apply_2d_rotary_pos_emb, SirenNet, apply_3d_rotary_pos_emb
from .basics import RMSNorm
import math

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class LowRankKernel(nn.Module):
    # low rank kernel, operates only on one dimension
    def __init__(self,
                 dim,
                 dim_head,
                 heads,
                 normalize=False,
                 softmax=False,
                 residual=True,
                 dropout=0,
                 scaling=1,
                 qk_norm=False,
                 normalized_to_one=False,
                 project_qry=True,
                 ):
        super().__init__()

        self.dim_head = dim_head
        self.heads = heads
        self.normalize = normalize
        self.residual = residual
        if dropout > 1e-6:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.to_q = nn.Linear(dim, dim_head*heads, bias=False) if project_qry else nn.Identity()
        self.to_k = nn.Linear(dim, dim_head*heads, bias=False)
        self.qk_norm = qk_norm
        if self.qk_norm:
            if self.normalize:
                raise ValueError('Cannot use qk_norm and normalize at the same time')
            self.q_norm = RMSNorm(dim_head, affine=True)
            self.k_norm = RMSNorm(dim_head, affine=False)
        self.normalize_to_one = normalized_to_one

        #self.init_gain = 0.02
        #self.initialize_qk_weights()
        self.softmax = softmax
        if self.normalize_to_one and self.use_softmax:
            raise ValueError('Cannot use softmax and normalize_to_one at the same time')

        self.residual = residual
        if self.residual:
            self.gamma = nn.Parameter(torch.tensor(1 / np.sqrt(dim_head)), requires_grad=True)
        else:
            self.gamma = 0
        self.scaling = scaling

    def initialize_qk_weights(self):
        xavier_uniform_(self.to_q.weight, gain=self.init_gain)
        xavier_uniform_(self.to_k.weight, gain=self.init_gain)

    def normalize_wrt_domain(self, x):
        # assuming domain is the second last
        x = (x - x.mean(dim=-2, keepdim=True)) / (x.std(dim=-2, keepdim=True) + 1e-5)
        return x

    def forward(self,
                u_x,
                pos_x=None,
                u_y=None,
                pos_y=None,
                rope_module=None,
                modulation=None):
        # u_x, u_y: b n c
        # u_x is from the first source
        # u_y is from the second source
        # pos: b n d
        # modulation: b n n c
        if u_y is None:
            u_y = u_x

        n = u_y.shape[1]

        q = self.to_q(u_x)
        k = self.to_k(u_y)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        if self.normalize:
            q = self.normalize_wrt_domain(q)
            k = self.normalize_wrt_domain(k)
        elif self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope_module is not None:
            q_freqs = rope_module(pos_x.squeeze())
            if pos_y is None:
                k_freqs = q_freqs
            else:
                k_freqs = rope_module(pos_y.squeeze())
            q_freqs = rearrange(q_freqs, 'n (h d) -> h n d', h=self.heads).unsqueeze(0)
            k_freqs = rearrange(k_freqs, 'n (h d) -> h n d', h=self.heads).unsqueeze(0)
            q = apply_rotary_pos_emb(q, q_freqs)
            k = apply_rotary_pos_emb(k, k_freqs)

        if modulation is not None:
            K = torch.einsum('bhid,bhjd,hijd->bhij', q, k, modulation) * self.scaling
        else:
            K = torch.einsum('bhid,bhjd->bhij', q, k) * self.scaling

        if self.normalize_to_one:
            K = K / K.sum(dim=-1, keepdim=True)

        K = self.dropout(K)
        if self.residual:
            K = K + self.gamma * torch.eye(n).to(q.device).view(1, 1, n, n)
        if self.softmax:
            K = F.softmax(K, dim=-1)

        return K











