import os
import torch
import numpy as np
from einops import rearrange, repeat
from torch.utils.data import Dataset
import glob

DEFAULT_FEATURES = [
    'geopotential',
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    'specific_humidity',
    '2m_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'mean_sea_level_pressure',
    'total_precipitation_6hr',
    'total_precipitation_24hr',
    '10m_wind_speed',
    'wind_speed'
]

DEFAULT_CONSTANTS = [
    'land_sea_mask',
    'soil_type',
]


class Normalizer:
    def __init__(self, stat_dict):
        # stat_dict: {feature_name: (mean, std)}
        self.stat_dict = {k: (torch.from_numpy(v[0]).float(), torch.from_numpy(v[1]).float()) if isinstance(v[0], np.ndarray)
        else (torch.tensor(v[0]).float(), torch.tensor(v[1]).float())
                          for k, v in stat_dict.items()}

    def normalize(self, out_dict):
        for k, v in out_dict.items():
            mean, std = self.stat_dict[k]
            mean, std = mean.to(v.device), std.to(v.device)
            if len(v.shape) == 3:
                out_dict[k] = (v - mean.view(1, 1, 1)) / std.view(1, 1, 1)
            elif len(v.shape) == 4:
                out_dict[k] = (v - mean.view(1, 1, 1, -1)) / std.view(1, 1, 1, -1)
            else:
                raise ValueError(f'Invalid shape {v.shape}')

    def denormalize(self, out_dict):
        for k, v in out_dict.items():
            mean, std = self.stat_dict[k]
            mean, std = mean.to(v.device), std.to(v.device)
            if len(v.shape) == 3:   # t nlat nlon
                out_dict[k] = v * std.view(1, 1, 1) + mean.view(1, 1, 1)
            elif len(v.shape) == 4:         # t nlat nlon nlevels
                out_dict[k] = v * std.view(1, 1, 1, -1) + mean.view(1, 1, 1, -1)
            else:
                raise ValueError(f'Invalid shape {v.shape}')

    def batch_denormalize(self, out_dict):
        for k, v in out_dict.items():
            mean, std = self.stat_dict[k]
            mean, std = mean.to(v.device), std.to(v.device)
            if len(v.shape) == 4:    # b t nlat nlon
                out_dict[k] = v * std.view(1, 1, 1, 1) + mean.view(1, 1, 1, 1)
            elif len(v.shape) == 5:     # b t nlat nlon nlevels
                out_dict[k] = v * std.view(1, 1, 1, 1, -1) + mean.view(1, 1, 1, 1, -1)
            else:
                raise ValueError(f'Invalid shape {v.shape}')

    def batch_normalize(self, out_dict):
        for k, v in out_dict.items():
            mean, std = self.stat_dict[k]
            mean, std = mean.to(v.device), std.to(v.device)
            if len(v.shape) == 4:    # b t nlat nlon
                out_dict[k] = (v - mean.view(1, 1, 1, 1)) / std.view(1, 1, 1, 1)
            elif len(v.shape) == 5:     # b t nlat nlon nlevels
                out_dict[k] = (v - mean.view(1, 1, 1, 1, -1)) / std.view(1, 1, 1, 1, -1)
            else:
                raise ValueError(f'Invalid shape {v.shape}')


class ResidualNormalizer:
    def __init__(self,
                 surface_bias, surface_scaling,
                 multi_level_bias, multi_level_scaling):
        self.surface_bias = surface_bias
        self.surface_scaling = surface_scaling
        self.multi_level_bias = multi_level_bias
        self.multi_level_scaling = multi_level_scaling

    def to(self, device):
        self.surface_bias = self.surface_bias.to(device)
        self.surface_scaling = self.surface_scaling.to(device)
        self.multi_level_bias = self.multi_level_bias.to(device)
        self.multi_level_scaling = self.multi_level_scaling.to(device)
        return self

    def scale_and_offset(self, surface_residual, multi_level_residual):
        surface_residual = surface_residual * self.surface_scaling + self.surface_bias
        multi_level_residual = multi_level_residual * self.multi_level_scaling + self.multi_level_bias
        return surface_residual, multi_level_residual


class ERA5Base:
    def __init__(self,
                 data_dir,
                 features_names,
                 constant_names,
                 feature_levels,
                 split='train',
                 interval=1,  # interval=1 is equal to 6 hours
                 nsteps=2,   # spit out how many consecutive future sequences
                 start_year=None,  # if not None, filter out data before start_year
                 ):

        self.data_dir = data_dir
        self.features_names = features_names
        self.feature_levels = feature_levels
        self.split = split
        assert self.split in ['train', 'valid']
        self.normalizer = Normalizer(self.load_norm_stats())

        # load constant
        self.constants = self.load_constant(constant_names)

        self.interval = interval
        self.nsteps = nsteps

        data_dir = os.path.join(data_dir, split)
        f_lst = glob.glob(os.path.join(data_dir, 'year_*'))
        f_lst = sorted(f_lst, key=lambda x: int(x.split('_')[-1]))
        # count how many years, get oldest and newest year
        self.nyears = [int(f.split('_')[-1]) for f in f_lst]
        if start_year is not None:
            self.nyears = [year for year in self.nyears if year >= start_year]
        self.oldest_year = self.nyears[0]
        self.newest_year = self.nyears[-1]

        # count how many days
        self.ndays = []
        for f in f_lst:
            year = int(f.split('_')[-1])
            if year % 4 == 0:
                self.ndays += [366]
            else:
                self.ndays += [365]

        # nhours = 24 * 365
        nstamps = [4*nday for nday in self.ndays]
        # self.nstamps = [nstamp - self.interval * self.nsteps + 1 for nstamp in nstamps]
        self.nstamps = [nstamp if i != len(nstamps)-1 else nstamp - self.interval * self.nsteps + 1
                        for i, nstamp in enumerate(nstamps)]

        # create a prefix sum table so that we can query the day of year given a day
        self.stamp_of_year = np.cumsum([0] + self.nstamps)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f'Found {len(self.nyears)} years of data, ranging from {self.oldest_year} to {self.newest_year}')
            print('There is a total of {} days'.format(sum(self.ndays)))

    def load_norm_stats(self):
        norm_stat_path = os.path.join(self.data_dir, 'norm_stats.npz')
        stat_dict = {}
        with np.load(norm_stat_path, allow_pickle=True) as f:
            normalize_mean, normalize_std = f['normalize_mean'].item(), f['normalize_std'].item()
            for feature_name in self.features_names:
                assert feature_name in normalize_mean.keys(), f'{feature_name} not in {norm_stat_path}'
                stat_dict[feature_name] = (normalize_mean[feature_name], normalize_std[feature_name])
        return stat_dict

    def load_constant(self, constant_names):
        constants_path = os.path.join(self.data_dir, 'constants.npz')
        constants = []
        with np.load(constants_path, allow_pickle=True) as f:
            for constant_name in constant_names:
                assert constant_name in f, f'{constant_name} not in {constants_path}'
                constant = f[constant_name]

                # normalize the constant to [-1, 1]
                constant = (constant - constant.min())*2 / (constant.max() - constant.min()) - 1.0
                constant = np.transpose(constant, (1, 0))
                constants += [constant]
        return np.stack(constants, axis=-1)


# map-style dataset, deprecated
class WeatherData(ERA5Base, torch.utils.data.Dataset):
    def __init__(self,
                 data_dir,
                 features_names,
                 constant_names,
                 feature_levels,
                 split='train',
                 interval=1,  # interval=1 is equal to 6 hours
                 nsteps=2,  # spit out how many consecutive future sequences
                 start_year=None,  # if not None, filter out data before start_year
                 ):
        super().__init__(data_dir, features_names, constant_names, feature_levels, split, interval, nsteps, start_year)

    def __len__(self):
        return sum(self.nstamps)

    def __getitem__(self, idx):
        # which year? which snapshots?
        year_idx = np.searchsorted(self.stamp_of_year, idx, side='right') - 1
        stamp_idx = idx - self.stamp_of_year[year_idx]

        # fetch data
        path = os.path.join(self.data_dir, self.split, f'year_{self.nyears[year_idx]}')

        # load all the features, each feature is stored in .npy file
        feat_dict = {}
        for feat_name in self.features_names:
            feat = np.load(os.path.join(path, f'{feat_name}.npy'), mmap_mode='r')[
                   stamp_idx:stamp_idx + self.interval * self.nsteps:self.interval]
            feat = torch.from_numpy(feat.copy()).float()
            t = feat.shape[0]
            if len(feat.shape) == 4:  # t, levels, nlon, nlat
                feat = rearrange(feat, 't levels nlon nlat -> t nlat nlon levels')
            elif len(feat.shape) == 3:  # t, nlon, nlat
                feat = rearrange(feat, 't nlon nlat -> t nlat nlon')

            feat_dict[feat_name] = feat

        if self.split == 'train':
            self.normalizer.normalize(feat_dict)
            # group feature into surface variables and multi-level variables
            surface_feat = []
            multi_level_feat = []
            for i, feat_name in enumerate(self.features_names):
                if len(feat_dict[feat_name].shape) == 4:
                    multi_level_feat += [feat_dict[feat_name]]
                else:
                    surface_feat += [feat_dict[feat_name]]
            surface_feat = torch.stack(surface_feat, dim=-1)
            multi_level_feat = torch.stack(multi_level_feat, dim=-1)
            # print(surface_feat.shape, multi_level_feat.shape, self.constants.shape)
            return surface_feat[0], surface_feat[1:], multi_level_feat[0], multi_level_feat[1:],\
                   torch.from_numpy(self.constants).float()
        else:
            #  get input at stamp_idx, output at stamp_idx + 1:stamp_idx + 1 + self.nsteps
            input_dict = {}
            output_dict = {}
            for k, v in feat_dict.items():
                input_dict[k] = v[0]
                output_dict[k] = v[1:]
            return input_dict, output_dict, torch.from_numpy(self.constants).float()


class ERA5EvalBase:
    def __init__(self,
                 data_dir,
                 features_names,
                 constant_names,
                 feature_levels,
                 interval=1,  # interval=1 is equal to 6 hours
                 nsteps=2,   # spit out how many consecutive future sequences
                 start_time_limit_years=None,  # if not None, force the starting time must be in start_years (which is a list)
                 ):

        self.data_dir = data_dir
        self.features_names = features_names
        self.feature_levels = feature_levels

        self.normalizer = Normalizer(self.load_norm_stats())

        # load constant
        self.constants = self.load_constant(constant_names)

        self.interval = interval
        self.nsteps = nsteps

        data_dir = os.path.join(data_dir, 'test')
        f_lst = glob.glob(os.path.join(data_dir, 'year_*'))
        f_lst = sorted(f_lst, key=lambda x: int(x.split('_')[-1]))
        # count how many years, get oldest and newest year
        self.nyears = [int(f.split('_')[-1]) for f in f_lst]  # this is a year list
        if start_time_limit_years is not None:
            self.start_time_limit_years = start_time_limit_years
        self.oldest_year = self.nyears[0]
        self.newest_year = self.nyears[-1]

        # count how many days
        self.ndays = 0

        for f in f_lst:
            year = int(f.split('_')[-1])
            if year % 4 == 0:
                self.ndays += 366
            else:
                self.ndays += 365

        # nhours = 24 * 365
        # every 6 hours
        self.nstamps = 4 * self.ndays

        # count how many days in each year and create a prefix sum table
        # so that we can query the day of year given a day
        self.day_of_year = np.cumsum([0] + [366 if year % 4 == 0 else 365 for year in self.nyears])

        print(f'Found {len(self.nyears)} years of data, ranging from {self.oldest_year} to {self.newest_year}')
        print('There is a total of {} days'.format(self.ndays))

    def load_norm_stats(self):
        norm_stat_path = os.path.join(self.data_dir, 'norm_stats.npz')
        stat_dict = {}
        with np.load(norm_stat_path, allow_pickle=True) as f:
            normalize_mean, normalize_std = f['normalize_mean'].item(), f['normalize_std'].item()
            for feature_name in self.features_names:
                assert feature_name in normalize_mean.keys(), f'{feature_name} not in {norm_stat_path}'
                stat_dict[feature_name] = (normalize_mean[feature_name], normalize_std[feature_name])
        return stat_dict

    def load_constant(self, constant_names):
        constants_path = os.path.join(self.data_dir, 'constants.npz')
        constants = []
        with np.load(constants_path, allow_pickle=True) as f:
            for constant_name in constant_names:
                assert constant_name in f, f'{constant_name} not in {constants_path}'
                constant = f[constant_name]

                # normalize the constant to [-1, 1]
                constant = (constant - constant.min())*2 / (constant.max() - constant.min()) - 1.0
                constant = np.transpose(constant, (1, 0))
                constants += [constant]
        return np.stack(constants, axis=-1)

