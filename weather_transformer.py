import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from libs.factorization_module import FABlockS2, FCABlockS2, CABlock
from libs.spherical_harmonics import SphericalHarmonicsPE
from libs.basics import MLP, PreNorm
import math
from libs.basics import GroupNorm
from libs.patchify_module import PatchifyConv, UnpatchifyEmbd, HeightUpsampleLayer

import abc


class FeatureEncoder(nn.Module):
    def __init__(self, multi_level_dim_in, surface_dim_in, dim, levels=1):
        super().__init__()
        self.multi_level_dim_in = multi_level_dim_in
        self.surface_dim_in = surface_dim_in
        self.dim = dim
        self.levels = levels

        self.depth_mixer = nn.Sequential(
            nn.Conv1d(multi_level_dim_in, dim, 1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=levels, stride=levels, padding=0, bias=False)
        )
        self.surface_mixer = nn.Linear(surface_dim_in, dim)

        self.channel_mixer = MLP([dim+dim, dim, dim], nn.GELU(), no_bias=False)

    def forward(self, x_surface, x_multi_level):
        # x_surface: b nlat nlon c
        # x_multi_level: b nlat nlon l c
        b, nlat, nlon = x_surface.shape[:3]
        z_surface = self.surface_mixer(x_surface)
        x_multi_level = rearrange(x_multi_level, 'b nlat nlon l c -> (b nlat nlon) c l')
        z_multi_level = self.depth_mixer(x_multi_level)
        z_multi_level = rearrange(z_multi_level, '(b nlat nlon) c 1 -> b nlat nlon c', b=b, nlat=nlat, nlon=nlon)
        z = torch.cat([z_surface, z_multi_level], dim=-1)
        return self.channel_mixer(z)


# flattened architecture
class FactFormerS2(nn.Module):
    def __init__(self,
                 dim,
                 dim_head,
                 latent_dim,
                 heads,
                 depth,    # number of blocks
                 **kwargs
                 ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.latent_dim = latent_dim
        self.heads = heads
        self.depth = depth
        self.layers = nn.ModuleList([])
        for i in range(depth):
            layer = nn.ModuleList([])
            layer.append(FABlockS2(dim, dim_head, latent_dim, heads, dim, **kwargs))

            # post norm setting
            if i != depth - 1:
                norm_layer = nn.LayerNorm(dim)
                layer.append(norm_layer)
            else:
                layer.append(nn.Identity())

            self.layers.append(layer)

    def forward(self, x, pos_lst, pe_lst=None):
        # x: b nlat nlon c
        for l, [attn, ln] in enumerate(self.layers):
            if pe_lst is None:
                x = attn(x, pos_lst) + x
                x = ln(x)
            else:
                x = attn(x+pe_lst[l], pos_lst) + x
                x = ln(x)
            #x = ffn(x) + x

        return x

    def get_attention_kernel(self, x, pos_lst, layer_idx, pe_lst=None):
        for l, [attn, ln] in enumerate(self.layers):
            if l != layer_idx:
                if pe_lst is None:
                    x = attn(x, pos_lst) + x
                    x = ln(x)
                else:
                    x = attn(x + pe_lst[l], pos_lst) + x
                    x = ln(x)
            else:
                if pe_lst is None:
                    k1, k2 = attn.get_kernel(x, pos_lst)
                else:
                    k1, k2 = attn.get_kernel(x + pe_lst[l], pos_lst)
                return k1, k2
        return None, None


class ThreedFeatureDecoder(nn.Module):
    def __init__(self, dim_in, dim, dim_out,
                 out_levels,
                 base_levels=1):
        super().__init__()
        self.dim_in = dim_in
        self.dim = dim
        self.dim_out = dim_out
        self.out_levels = out_levels
        self.base_levels = base_levels
        if out_levels != 1:
            self.depth_upsample = nn.Sequential(
                GroupNorm(32, dim_in),
                HeightUpsampleLayer(dim_in, dim//2, base_levels, dim, out_levels))

            self.channel_mixer = nn.Sequential(
                nn.GELU(),
                nn.Linear(dim, dim_out))
        else:
            # not upsample anything in fact
            self.depth_upsample = nn.Sequential(
                                  GroupNorm(32, dim_in),
                                  MLP([dim_in, dim, dim], nn.GELU()))

            self.channel_mixer = nn.Sequential(
                GroupNorm(32, dim),
                nn.GELU(),
                nn.Linear(dim, dim_out))

    def forward(self, x):
        # b, nlat, nlon, c = x.shape
        x = self.depth_upsample(x)
        x = self.channel_mixer(x)
        return x


class ClimaAutoencoder(nn.Module):
    def __init__(self,
                 config):

        super().__init__()
        # retrieve model configs
        self.base_dim = config.model.base_dim
        self.latent_dim = config.model.latent_dim

        self.ae_dim_head = config.model.decoder.dim_head
        self.ae_latent_dim = config.model.decoder.latent_dim
        self.ae_heads = config.model.decoder.heads

        self.ae_kernel_multiplier = config.model.decoder.kernel_multiplier
        self.ae_use_pe = config.model.decoder.use_distance_encoding
        self.ae_use_softmax = config.model.decoder.use_softmax
        self.ae_qk_norm = config.model.decoder.qk_norm

        self.l_spherical_harmonics = config.model.l_spherical_harmonics

        # variables that are strongly correlated with each other are grouped together
        self.constant_name = config.data.constant_names   # n_constants
        self.variable_groups = config.data.variable_groups   # n_groups
        self.variable_name = [] # n_variables
        for vars in self.variable_groups:
            self.variable_name.extend(vars)
        # 1 for surface, >1 for multi-level, assuming all multi-level variables have the same number of levels
        # variables of different levels cannot be grouped together
        self.variable_levels = config.data.variable_levels   # n_variables
        # run a check to see if the variables are grouped correctly
        for vars, level in zip(self.variable_groups, self.variable_levels):
            # assert all var in vars have same level
            assert len(set([self.variable_levels[self.variable_name.index(var)] for var in vars])) == 1, \
                'Variables in the same group must have the same number of levels'

        self.n_variables = len(self.variable_name)
        self.n_groups = len(self.variable_groups)
        self.n_levels = sorted(self.variable_levels)[-1]

        # self.pivot_levels = config.model.pivot_levels

        multi_level_variables = [n for n, l in zip(self.variable_name, self.variable_levels) if l > 1]
        surface_variables = [n for n, l in zip(self.variable_name, self.variable_levels) if l == 1]

        self.feature_encoder = FeatureEncoder(multi_level_dim_in=len(multi_level_variables),
                                              surface_dim_in=len(surface_variables) + len(self.constant_name),
                                              dim=self.base_dim,
                                              levels=self.n_levels)

        self.ca_decoder = CABlock(self.latent_dim,
                                  self.base_dim,
                                  self.ae_dim_head,
                                  self.ae_latent_dim,
                                  self.ae_heads,
                                  use_pe=True,
                                  kernel_multiplier=self.ae_kernel_multiplier,
                                  use_softmax=self.ae_use_softmax,
                                  qk_norm=self.ae_qk_norm)

        self.nlon, self.nlat = config.data.nlon, config.data.nlat

        self.register_buffer('latitude', torch.linspace(-math.pi / 2, math.pi / 2, self.nlat), persistent=False)
        self.register_buffer('longitude', torch.linspace(0, 2 * math.pi - (2 * math.pi / self.nlon), self.nlon),
                             persistent=False)

        self.pivot_ratio = config.model.pivot_ratio

        if self.nlat % self.pivot_ratio == 0:
            pivot_row = torch.arange(1, self.nlat, self.pivot_ratio)
        else:
            pivot_row = torch.arange(0, self.nlat, self.pivot_ratio)
        self.register_buffer('pivot_latitude', self.latitude[pivot_row], persistent=False)

        pivot_col = torch.arange(1, self.nlon, self.pivot_ratio)
        self.register_buffer('pivot_longitude', self.longitude[pivot_col], persistent=False)

        self.patch_embd = PatchifyConv(self.pivot_ratio, self.base_dim, self.latent_dim)

        self.spherical_pe_orig = SphericalHarmonicsPE(self.l_spherical_harmonics, self.base_dim, self.base_dim)
        self.spherical_pe_orig.cache_precomputed_sph_harmonics(self.latitude + math.pi / 2., self.longitude)

        self.spherical_pe_pivot = SphericalHarmonicsPE(self.l_spherical_harmonics, self.latent_dim, self.latent_dim,
                                                         use_mlp=False)
        self.spherical_pe_pivot.cache_precomputed_sph_harmonics(self.pivot_latitude + math.pi / 2.,
                                                                 self.pivot_longitude)

        # for the feature decoder, we need to first split the features into different groups
        self.feature_decoder_dict = nn.ModuleDict({})
        for i, group in enumerate(self.variable_groups):
            group_id = 'group_' + str(i)
            level_num = self.variable_levels[self.variable_name.index(group[0])]
            self.feature_decoder_dict[group_id] = ThreedFeatureDecoder(dim_in=self.base_dim,
                                                                       dim=self.base_dim,
                                                                       dim_out=len(group),
                                                                       out_levels=level_num)

    def get_pe(self):
        pe_orig = self.spherical_pe_orig(self.latitude, self.longitude)
        # pe_encode, pe_decode = torch.chunk(pe_orig, 2, dim=-1)
        pe_encode = pe_orig
        pe_decode = 0
        pe_pivot_raw = self.spherical_pe_pivot(self.pivot_latitude, self.pivot_longitude)
        return pe_encode, pe_decode, pe_pivot_raw

    def encode(self, surface, multi_level, constant, pe_encode):
        surface_feat = torch.cat([surface, constant], dim=-1)

        z = self.feature_encoder(surface_feat, multi_level)   # [b, l, nlat, nlon, c]
        z += pe_encode

        z = self.patch_embd(z)
        return z

    def decode(self, z, pe_decode):
        z = self.ca_decoder(z, [self.latitude, self.longitude], [self.pivot_latitude, self.pivot_longitude])

        # check if the output is of the same size as the specified resolution, if larger, unpad them
        nlat, nlon = self.latitude.shape[0], self.longitude.shape[0]
        b, h, w, c = z.shape
        if h > nlat:
            z = z[:, :nlat, :, :]
        if w > nlon:
            z = z[:, :, :nlon, :]

        surface_feat_out = []
        multi_level_feat_out = []
        for i, group in enumerate(self.variable_groups):
            group_id = 'group_' + str(i)
            out_feat = self.feature_decoder_dict[group_id](z)

            if self.variable_levels[self.variable_name.index(group[0])] == 1:
                surface_feat_out.append(out_feat)
            else:
                multi_level_feat_out.append(out_feat)
        surface_feat_out = torch.cat(surface_feat_out, dim=-1)
        multi_level_feat_out = torch.cat(multi_level_feat_out, dim=-1)

        return surface_feat_out, multi_level_feat_out

    def forward(self, surface_feat_in, multi_level_feat_in, constant):
        pe_encode, pe_decode, pe_pivot_raw = self.get_pe()   # pe_pivot_raw is not used in the forward pass
        z = self.encode(surface_feat_in, multi_level_feat_in, constant, pe_encode)
        surface_feat_out, multi_level_feat_out = self.decode(z, pe_decode)

        return surface_feat_out, multi_level_feat_out


class CaFABase(nn.Module):
    def __init__(self,
                 config):

        super().__init__()
        # retrieve model configs
        self.base_dim = config.model.base_dim
        self.latent_dim = config.model.latent_dim

        self.autoencoder = ClimaAutoencoder(config)

        self.processor_dim_head = config.model.processor.dim_head
        self.processor_latent_dim = config.model.processor.latent_dim
        self.processor_heads = config.model.processor.heads
        self.processor_kernel_multiplier = config.model.processor.kernel_multiplier
        self.processor_use_pe = config.model.processor.use_distance_encoding
        self.processor_use_softmax = config.model.processor.use_softmax
        self.processor_qk_norm = config.model.processor.qk_norm
        self.processor_depth = config.model.processor.depth

        self.l_spherical_harmonics = config.model.l_spherical_harmonics
        self.pivot_ratio = config.model.pivot_ratio

        self.processor = FactFormerS2(self.latent_dim,
                                      self.processor_dim_head,
                                      self.processor_latent_dim,
                                      self.processor_heads,
                                      depth=self.processor_depth,
                                      use_pe=self.processor_use_pe,
                                      kernel_multiplier=self.processor_kernel_multiplier,
                                      use_softmax=self.processor_use_softmax,
                                      qk_norm=self.processor_qk_norm,
                                      zero_init=True)

        self.processor_pe_mlp = MLP([self.latent_dim,
                                 self.latent_dim*2, self.latent_dim*self.processor_depth], nn.GELU(), no_bias=False)

    def load_pretrained_ae(self, path):
        self.autoencoder.load_state_dict(torch.load(path)['model'])

    def process(self, z, pe_process):
        z = self.processor(z, [self.autoencoder.pivot_latitude, self.autoencoder.pivot_longitude], pe_process)
        return z

    # to be implemented in subclass
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError


# CaFA with encoder-processor-decoder scheme
class CaFAEPD(CaFABase):
    def __init__(self,
                 config):

        super().__init__(config)

    def forward(self,
                surface_feat_in, multi_level_feat_in, constant):
        # surface_feat: b x nlat x nlon x nvar
        # multi_level_feat: b x nlat x nlon x levels x nvar
        # constant: b x nlat x nlon x n_constants
        # in processor, the pe need to sampled according to the pivot

        b, nlat, nlon, c = surface_feat_in.shape

        pe_encode, pe_decode, pe_pivot_raw = self.autoencoder.get_pe()
        z = self.autoencoder.encode(surface_feat_in, multi_level_feat_in, constant, pe_encode)

        pe_process = self.processor_pe_mlp(pe_pivot_raw)
        pe_process_lst = torch.chunk(pe_process, chunks=self.processor_depth, dim=-1)
        z = self.process(z, pe_process_lst)

        surface_feat_out, multi_level_feat_out = self.autoencoder.decode(z, pe_decode)

        # residual style prediction

        return surface_feat_out, multi_level_feat_out












