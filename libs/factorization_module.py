import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from libs.positional_encoding_module import RadialBesselBasis, SirenNet
from libs.basics import PreNorm, PostNorm, GeAct, MLP, GroupNorm, InstanceNorm, RMSNorm
from libs.attention import LowRankKernel


class PoolingReducer(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim):
        super().__init__()
        self.to_in = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_ffn = PreNorm(in_dim, MLP([hidden_dim, hidden_dim, out_dim], GeAct(nn.GELU())))

    def forward(self, x, mesh_weights=None):
        # note that the dimension to be pooled will be the last dimension
        # x: b nx ... c
        # mesh_weights: nx
        x = self.to_in(x)
        # pool all spatial dimension but the first one
        ndim = len(x.shape)
        if mesh_weights is not None:
            # mesh_weights: nx
            # x: b nx ny nz ... c
            x = torch.einsum('b n ... c, n -> b n ... c', x, mesh_weights)
        x = x.mean(dim=tuple(range(2, ndim-1)))
        x = self.out_ffn(x)
        return x  # b nx c


class FABlockS2(nn.Module):
    # contains factorization and attention on each axis (latitute and longitude)
    def __init__(self,
                 dim,
                 dim_head,
                 latent_dim,
                 heads,
                 dim_out,
                 use_pe=True,
                 kernel_multiplier=3,
                 scaling_factor=1.0,
                 dropout=0.0,
                 qk_norm=False,
                 use_softmax=False,
                 polar_eps=1,
                 zero_init=False):
        super().__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.heads = heads
        self.dim_head = dim_head
        self.use_softmax = use_softmax
        self.polar_eps = polar_eps

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.channel_mixer = MLP([dim, dim*6, heads * dim_head + dim*2], nn.GELU())

        self.to_long = PoolingReducer(self.dim, self.dim, self.latent_dim)
        self.to_lat = PoolingReducer(self.dim, self.dim, self.latent_dim)

        self.low_rank_kernel_long = LowRankKernel(self.latent_dim, dim_head * kernel_multiplier, heads,
                                               residual=False,  # add a diagonal bias
                                               softmax=use_softmax,
                                               scaling=1 / np.sqrt(dim_head * kernel_multiplier)
                                               if kernel_multiplier > 4 or use_softmax else scaling_factor,
                                               qk_norm=qk_norm)
        self.low_rank_kernel_lat = LowRankKernel(self.latent_dim, dim_head * kernel_multiplier, heads,
                                               residual=False,
                                               softmax=use_softmax,
                                               scaling=1 / np.sqrt(dim_head * kernel_multiplier)
                                               if kernel_multiplier > 4 or use_softmax else scaling_factor,
                                               qk_norm=qk_norm)
        self.use_pe = use_pe
        if use_pe:
            self.pe_long = RadialBesselBasis(num_kernels=64, num_heads=heads, dim_head=dim_head * kernel_multiplier,
                                             enforce_periodicity=True)
            self.pe_lat = RadialBesselBasis(num_kernels=32, num_heads=heads, dim_head=dim_head * kernel_multiplier,
                                            enforce_periodicity=False)

        self.merge_head = nn.Sequential(
            GroupNorm(heads, dim_head*heads),
            nn.Linear(dim_head * heads, dim_out, bias=False))

        self.to_out = nn.Linear(dim_out, dim_out, bias=False)
        if zero_init:
            nn.init.zeros_(self.to_out.weight)

    def get_latitude_weights(self, lat):
        weights_lat = torch.cos(lat)  # nlat
        # add eps if both lat coords contain +/- 90 degree
        if weights_lat[0] < 1e-3:
            dlat = lat[1] - lat[0]
            polar_weight = torch.sin(dlat / 4) ** 2 / torch.sin(dlat / 2)
            weights_lat[0] = self.polar_eps * polar_weight
        if weights_lat[-1] < 1e-3:
            dlat = lat[1] - lat[0]
            polar_weight = torch.sin(dlat / 4) ** 2 / torch.sin(dlat / 2)
            weights_lat[-1] = self.polar_eps * polar_weight
        return weights_lat

    def forward(self, u, pos_lst,
                cache_pos=True,   # assume position doesn't change, we can cache the distance matrix
                ):

        u = self.channel_mixer(u)
        v, u, u_skip = torch.split(u, [self.heads * self.dim_head, self.dim, self.dim], dim=-1)

        lat, long = pos_lst
        if cache_pos and not hasattr(self, 'weights_lat'):
            weights_lat = self.get_latitude_weights(lat)
            self.weights_lat = weights_lat.detach()
        elif cache_pos and hasattr(self, 'weights_lat'):
            weights_lat = self.weights_lat
        else:
            weights_lat = self.get_latitude_weights(lat)

        # weights_lat /= weights_lat.mean()
        n_lat, n_lon = lat.shape[-1], long.shape[-1]
        u_long = self.to_long(rearrange(u, 'b nlat nlon c -> b nlon nlat c'))
        u_lat = self.to_lat(u, weights_lat)

        # apply distance decay
        if self.use_pe:
            if cache_pos and not hasattr(self, 'dist_cache'):
                long_dist_mat = abs(long.unsqueeze(0) - long.unsqueeze(1))
                lat_dist_mat = abs(lat.unsqueeze(0) - lat.unsqueeze(1))
                self.dist_cache = (long_dist_mat.detach(), lat_dist_mat.detach())
            elif cache_pos and hasattr(self, 'dist_cache'):
                long_dist_mat, lat_dist_mat = self.dist_cache
            else:
                long_dist_mat = abs(long.unsqueeze(0) - long.unsqueeze(1))
                lat_dist_mat = abs(lat.unsqueeze(0) - lat.unsqueeze(1))

            long_dist_decay = self.pe_long(long_dist_mat)  # heads x nlong x nlong
            lat_dist_decay = self.pe_lat(lat_dist_mat)  # heads x nlat x nlat

            k_long = self.low_rank_kernel_long(u_long, pos_x=long, modulation=long_dist_decay)
            k_lat = self.low_rank_kernel_lat(u_lat, pos_x=lat, modulation=lat_dist_decay)

        else:
            k_long = self.low_rank_kernel_long(u_long, pos_x=long)
            k_lat = self.low_rank_kernel_lat(u_lat, pos_x=lat)

        # add gating
        if not self.use_softmax:
            k_long = F.leaky_relu_(k_long, 0.2)
            k_lat = F.leaky_relu_(k_lat, 0.2)

        u_phi = rearrange(v, 'b i l (h c) -> b h i l c', h=self.heads)
        k_lat = torch.einsum('bhij,j->bhij', k_lat, weights_lat)
        u_phi = torch.einsum('bhij,bhjmc->bhimc', k_lat, u_phi) * np.pi / n_lat
        u_phi = torch.einsum('bhlm,bhimc->bhilc', k_long, u_phi) * (2 * np.pi / n_lon)
        u_phi = rearrange(u_phi, 'b h i l c -> b i l (h c)', h=self.heads)
        u_phi = self.merge_head(u_phi)
        return self.to_out(u_phi+u_skip)

    @torch.no_grad()
    def get_kernel(self, u, pos_lst,):   # get the axial kernel
        u = self.channel_mixer(u)
        v, u, u_skip = torch.split(u, [self.heads * self.dim_head, self.dim, self.dim], dim=-1)

        lat, long = pos_lst

        weights_lat = self.get_latitude_weights(lat)

        # weights_lat /= weights_lat.mean()
        n_lat, n_lon = lat.shape[-1], long.shape[-1]
        u_long = self.to_long(rearrange(u, 'b nlat nlon c -> b nlon nlat c'))
        u_lat = self.to_lat(u, weights_lat)

        # apply distance decay
        if self.use_pe:
            long_dist_mat = abs(long.unsqueeze(0) - long.unsqueeze(1))
            lat_dist_mat = abs(lat.unsqueeze(0) - lat.unsqueeze(1))

            long_dist_decay = self.pe_long(long_dist_mat)  # heads x nlong x nlong
            lat_dist_decay = self.pe_lat(lat_dist_mat)  # heads x nlat x nlat

            k_long = self.low_rank_kernel_long(u_long, pos_x=long, modulation=long_dist_decay)
            k_lat = self.low_rank_kernel_lat(u_lat, pos_x=lat, modulation=lat_dist_decay)

        else:
            k_long = self.low_rank_kernel_long(u_long, pos_x=long)
            k_lat = self.low_rank_kernel_lat(u_lat, pos_x=lat)

        # add gating
        if not self.use_softmax:
            k_long = F.leaky_relu_(k_long, 0.2)
            k_lat = F.leaky_relu_(k_lat, 0.2)

        return k_lat, k_long


class FCABlockS2(nn.Module):
    # conduct cross-attention with factorized attention kernels
    def __init__(self,
                 dim,
                 dim_head,
                 latent_dim,
                 heads,
                 dim_out,
                 use_pe=True,
                 kernel_multiplier=3,
                 scaling_factor=1.0,
                 dropout=0.0,
                 qk_norm=False,
                 use_softmax=False,
                 polar_eps=1,
                 **kwargs):
        super().__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.heads = heads
        self.dim_head = dim_head
        self.use_softmax = use_softmax
        self.polar_eps = polar_eps

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.channel_mixer = nn.Sequential(
            GroupNorm(32, dim),
            MLP([dim, dim*4, heads * dim_head + dim], nn.GELU()))

        # this will be used to project the src to key
        self.to_long = PoolingReducer(self.dim, self.dim, self.latent_dim)
        self.to_lat = PoolingReducer(self.dim, self.dim, self.latent_dim)

        self.low_rank_kernel_long = LowRankKernel(self.latent_dim, dim_head * kernel_multiplier, heads,
                                               residual=False,  # add a diagonal bias
                                               softmax=use_softmax,
                                               scaling=1 / np.sqrt(dim_head * kernel_multiplier)
                                               if kernel_multiplier > 4 or use_softmax else scaling_factor,
                                               qk_norm=qk_norm,
                                               project_qry=False)
        self.low_rank_kernel_lat = LowRankKernel(self.latent_dim, dim_head * kernel_multiplier, heads,
                                               residual=False,
                                               softmax=use_softmax,
                                               scaling=1 / np.sqrt(dim_head * kernel_multiplier)
                                               if kernel_multiplier > 4 or use_softmax else scaling_factor,
                                               qk_norm=qk_norm,
                                                project_qry=False)
        self.use_pe = use_pe
        if use_pe:
            self.pe_long = RadialBesselBasis(num_kernels=64, num_heads=heads, dim_head=dim_head * kernel_multiplier,
                                             enforce_periodicity=True)
            self.pe_lat = RadialBesselBasis(num_kernels=32, num_heads=heads, dim_head=dim_head * kernel_multiplier,
                                            enforce_periodicity=False)

        self.merge_head = nn.Sequential(
            GroupNorm(heads, dim_head*heads),
            nn.Linear(dim_head * heads, dim_out, bias=False))

    def get_latitude_weights(self, lat):
        weights_lat = torch.cos(lat)  # nlat
        # add eps if both lat coords contain +/- 90 degree
        if weights_lat[0] < 1e-3:
            dlat = lat[1] - lat[0]
            polar_weight = torch.sin(dlat / 4) ** 2 / torch.sin(dlat / 2)
            weights_lat[0] = self.polar_eps * polar_weight
        if weights_lat[-1] < 1e-3:
            dlat = lat[1] - lat[0]
            polar_weight = torch.sin(dlat / 4) ** 2 / torch.sin(dlat / 2)
            weights_lat[-1] = self.polar_eps * polar_weight
        return weights_lat

    def forward(self, u_src, query_basis_lst, query_pos_lst, src_pos_lst, cache_pos=True):
        u_src = self.channel_mixer(u_src)
        v_src, u_src = torch.split(u_src, [self.heads * self.dim_head, self.dim], dim=-1)
        src_lat, src_long = src_pos_lst
        qry_lat, qry_long = query_pos_lst
        u_lat_qry, u_long_qry = query_basis_lst
        # print(u_lat_qry.shape, u_long_qry.shape)

        if cache_pos and not hasattr(self, 'weights_lat'):
            weights_lat = self.get_latitude_weights(src_lat)
            self.weights_lat = weights_lat.detach()
        elif cache_pos and hasattr(self, 'weights_lat'):
            weights_lat = self.weights_lat
        else:
            weights_lat = self.get_latitude_weights(src_lat)

        n_lat_src, n_lon_src = src_lat.shape[-1], src_long.shape[-1]
        u_long_src = self.to_long(rearrange(u_src, 'b nlat nlon c -> b nlon nlat c'))
        u_lat_src = self.to_lat(u_src, weights_lat)

        # apply distance decay
        if self.use_pe:
            # the distance matrix will be n_query x n_src for both lat and long
            if cache_pos and not hasattr(self, 'dist_cache'):
                long_dist_mat = abs(src_long.unsqueeze(0) - qry_long.unsqueeze(1))
                lat_dist_mat = abs(src_lat.unsqueeze(0) - qry_lat.unsqueeze(1))
                self.dist_cache = (long_dist_mat.detach(), lat_dist_mat.detach())
            elif cache_pos and hasattr(self, 'dist_cache'):
                long_dist_mat, lat_dist_mat = self.dist_cache
            else:
                long_dist_mat = abs(src_long.unsqueeze(0) - qry_long.unsqueeze(1))
                lat_dist_mat = abs(src_lat.unsqueeze(0) - qry_lat.unsqueeze(1))

            long_dist_decay = self.pe_long(long_dist_mat)  # heads x qry_nlong x src_nlong
            lat_dist_decay = self.pe_lat(lat_dist_mat)  # heads x qry_nlat x src_nlat
            # print(long_dist_decay.shape, lat_dist_decay.shape)
            k_long = self.low_rank_kernel_long(u_x=u_long_qry,
                                               u_y=u_long_src,
                                               modulation=long_dist_decay)
            k_lat = self.low_rank_kernel_lat(u_x=u_lat_qry,
                                             u_y=u_lat_src,
                                             modulation=lat_dist_decay
                                             )

        else:
            k_long = self.low_rank_kernel_long(u_x=u_long_qry,
                                               u_y=u_long_src)
            k_lat = self.low_rank_kernel_lat(u_x=u_lat_qry,
                                             u_y=u_lat_src)

        # add gating
        if not self.use_softmax:
            k_long = F.leaky_relu_(k_long, 0.2)
            k_lat = F.leaky_relu_(k_lat, 0.2)

        u_phi = rearrange(v_src, 'b i l (h c) -> b h i l c', h=self.heads)
        k_lat = torch.einsum('bhij,j->bhij', k_lat, weights_lat)
        u_phi = torch.einsum('bhij,bhjmc->bhimc', k_lat, u_phi) * (np.pi / n_lat_src)
        u_phi = torch.einsum('bhlm,bhimc->bhilc', k_long, u_phi) * (2 * np.pi / n_lon_src)
        u_phi = rearrange(u_phi, 'b h i l c -> b i l (h c)', h=self.heads)
        u_phi = self.merge_head(u_phi)
        return u_phi


# wrapper for factorized cross-attention block
class CABlock(nn.Module):
    def __init__(self,
                 dim,
                 dim_out,
                 dim_head,
                 latent_dim,
                 heads,
                 **kwargs):

        super().__init__()
        kernel_multiplier = kwargs.get('kernel_multiplier', 2)
        self.qry_basis_lat = SirenNet(1, dim*2, dim_head*heads*kernel_multiplier, 4)
        self.qry_basis_long = SirenNet(1, dim*2, dim_head*heads*kernel_multiplier, 4)

        self.attn_layer = FCABlockS2(dim, dim_head, latent_dim, heads, dim_out, **kwargs)

    def forward(self, src, qry_pos_lst, src_pos_lst):
        # src: b nlat nlon c
        # qry_pos_lst: [qry_latitude, qry_longitude]
        # src_pos_lst: [src_latitude, src_longitude]
        b, nlat, nlon, c = src.shape
        qry_lat, qry_long = qry_pos_lst

        # make sure qyr_pos are in [-1, 1] before feeding to siren
        qry_y = (qry_lat + np.pi/2) / np.pi - 0.5
        qry_x = qry_long / (2*np.pi) - 0.5

        q1 = self.qry_basis_lat(qry_y.unsqueeze(-1))  # [nlat_qry, dim_head*heads*kernel_multiplier]
        q2 = self.qry_basis_long(qry_x.unsqueeze(-1))  # [nlong_qry, dim_head*heads*kernel_multiplier]
        q1 = repeat(q1, 'nlat d -> b nlat d', b=b)
        q2 = repeat(q2, 'nlon d -> b nlon d', b=b)
        x = self.attn_layer(src, [q1, q2], qry_pos_lst, src_pos_lst)
        return x