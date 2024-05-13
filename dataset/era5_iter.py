import math
import os
import random
import glob

import numpy as np
import torch
from torch.utils.data import IterableDataset

from dataset.era5 import DEFAULT_FEATURES, DEFAULT_CONSTANTS, Normalizer, ERA5Base, ERA5EvalBase
from einops import rearrange, repeat


class ERA5Reader(ERA5Base, IterableDataset):
    def __init__(self,
                 data_dir,
                 features_names,
                 constant_names,
                 feature_levels,
                 split='train',
                 init_time='all',    # all or 0/12 or 6/18
                 interval=1,  # interval=1 is equal to 6 hours
                 nsteps=2,   # spit out how many consecutive future sequences
                 start_year=None,  # if not None, filter out data not in start_year
                 years_range=(-1, -1),  # if not (-1, -1), filter out data not in the range
                 shuffle: bool = False,
                 ):

        super().__init__(data_dir, features_names, constant_names, feature_levels, split, interval, nsteps, start_year)

        self.shuffle = shuffle
        # loop through all the snapshots
        self.year_lst = self.nyears[:]
        self.stamp_of_year_lst = self.nstamps[:]
        self.init_time = init_time
        if years_range[0] != -1 or years_range[1] != -1:
            assert years_range[0] <= years_range[1], 'Invalid years_range'
            self.stamp_of_year_lst = [s for i, s in enumerate(self.stamp_of_year_lst) if years_range[0] <= self.year_lst[i] <= years_range[1]]
            self.year_lst = [y for y in self.year_lst if years_range[0] <= y <= years_range[1]]
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(f'Filter out years not in the range: {years_range}')
                print(f'Original # of years: {len(self.nyears)}, Filtered # of years: {len(self.year_lst)}')

    def __iter__(self):
        if self.shuffle:
            shuffled_idx = list(range(len(self.year_lst)))
            random.shuffle(shuffled_idx)
            # sync the order of year_lst and stamp_of_year_lst
            self.year_lst = [self.year_lst[i] for i in shuffled_idx]
            self.stamp_of_year_lst = [self.stamp_of_year_lst[i] for i in shuffled_idx]
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None or self.split != 'train':
            iter_start = 0
            iter_end = len(self.year_lst)
        else:
            # currently this only shards the data for training and shards based on the year
            if not torch.distributed.is_initialized():
                rank = 0
                world_size = 1
            else:
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            num_workers_per_ddp = worker_info.num_workers
            num_shards = num_workers_per_ddp * world_size

            per_worker = int(math.floor(len(self.year_lst) / float(num_shards)))
            worker_id = rank * num_workers_per_ddp + worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker

        for idx in range(iter_start, iter_end):
            # which year? which snapshots?
            year = self.year_lst[idx]
            stamp_of_year = list(range(self.stamp_of_year_lst[idx]))

            # please note that ideally we should use all the stamp_of_year for training
            # this is to accomodate the protocol in WeatherBench paper
            # See page 5 of https://arxiv.org/pdf/2308.15560.pdf
            if self.init_time == '0/12':
                stamp_of_year = stamp_of_year[::2]
            elif self.init_time == '6/18':
                stamp_of_year = stamp_of_year[1::2]
            else:
                pass        # do nothing, use all the stamp

            year_data_dict = {}
            for feat_name in self.features_names:
                path = os.path.join(self.data_dir, self.split, f'year_{year}', feat_name + '.npy')
                feat = np.load(path)
                # if year is the last year in the year list, don't load next year
                if (year+1) in self.year_lst:
                    next_year_path = os.path.join(self.data_dir, self.split, f'year_{year+1}', feat_name + '.npy')
                    next_year_feat = np.load(next_year_path)
                    feat = np.concatenate([feat, next_year_feat], axis=0)
                t = feat.shape[0]
                if len(feat.shape) == 4:  # t, levels, nlon, nlat -> t, nlat, nlon, levels
                    feat = np.transpose(feat, (0, 3, 2, 1))
                elif len(feat.shape) == 3:  # t, nlon, nlat -> t, nlat, nlon
                    feat = np.transpose(feat, (0, 2, 1))
                year_data_dict[feat_name] = feat
            # shuffle it
            if self.shuffle:
                random.shuffle(stamp_of_year)
            for start_stamp_idx in stamp_of_year:
                end_stamp_idx = start_stamp_idx + self.nsteps*self.interval
                feat_dict = {k:
                             torch.from_numpy(v[start_stamp_idx:end_stamp_idx:self.interval]).float()
                             for k, v in year_data_dict.items()}
                start_dayofyear = start_stamp_idx // 4
                start_hour = (start_stamp_idx % 4) * 6

                # append time to the feature dict
                if self.split != 'train':
                    feat_dict['time_dayofyear'] = torch.tensor([start_dayofyear])
                    feat_dict['time_hour'] = torch.tensor([start_hour])
                    feat_dict['time_year'] = torch.tensor([year])

                yield feat_dict


class WeatherForecastData(IterableDataset):
    def __init__(self,
                 datareader):
        self.datareader = datareader
        super().__init__()

    def get_per_epoch_iters(self):
        # get the number of iterations per epoch
        return sum(self.datareader.nstamps)

    def __iter__(self):
        for data in self.datareader:
            # data is a dict of features
            if self.datareader.split == 'train':
                self.datareader.normalizer.normalize(data)
                # group feature into surface variables and multi-level variables
                surface_feat = []
                multi_level_feat = []
                for i, feat_name in enumerate(self.datareader.features_names):
                    if len(data[feat_name].shape) == 4:
                        multi_level_feat += [data[feat_name]]
                    else:
                        surface_feat += [data[feat_name]]
                surface_feat = torch.stack(surface_feat, dim=-1)
                multi_level_feat = torch.stack(multi_level_feat, dim=-1)
                # print(surface_feat.shape, multi_level_feat.shape, self.constants.shape)
                yield surface_feat[0], surface_feat[1:], multi_level_feat[0], multi_level_feat[1:], \
                       torch.from_numpy(self.datareader.constants).float()
            else:
                #  get input at stamp_idx, output at stamp_idx + 1:stamp_idx + 1 + self.nsteps
                input_dict = {}
                output_dict = {}
                for k, v in data.items():
                    if not k.startswith('time'):
                        input_dict[k] = v[0]
                        output_dict[k] = v[1:]
                yield input_dict, output_dict, torch.from_numpy(self.datareader.constants).float(), \
                        data['time_dayofyear'], data['time_hour'], data['time_year']


class ERA5EvalReader(ERA5EvalBase, IterableDataset):
    def __init__(self,
                 data_dir,
                 features_names,
                 constant_names,
                 feature_levels,
                 init_time='all',    # all or 0/12 or 6/18
                 interval=1,  # interval=1 is equal to 6 hours
                 nsteps=2,   # spit out how many consecutive future sequences
                 start_time_limit_years=[],  # if not None, force the starting time must be in start_years (which is a list)
                 ):
        super().__init__(data_dir, features_names, constant_names, feature_levels,
                         interval, nsteps,
                         start_time_limit_years)

        # loop through all the snapshots
        self.year_lst = self.nyears[:]
        self.split = 'test'
        self.init_time = init_time
        print(f'init_time: {init_time}')

    def __iter__(self):
        # do not shuffle
        iter_start = 0
        iter_end = len(self.year_lst)

        for idx in range(iter_start, iter_end):  # loop through all the days
            year_idx = self.year_lst[idx]
            # which year? which snapshots?
            if self.start_time_limit_years and \
                    not year_idx in self.start_time_limit_years:
                continue
            days_in_year = 365 if year_idx % 4 != 0 else 366

            # every day has 4 starting snapshots
            if self.init_time == '0/12':
                stamp_of_year = np.arange(0, days_in_year*4, 2)
            elif self.init_time == '6/18':
                stamp_of_year = np.arange(1, days_in_year*4, 2)
            elif self.init_time == 'all':
                stamp_of_year = np.arange(days_in_year*4)        # use all the stamp
            else:
                raise ValueError(f'Unknown init_time: {self.init_time}')

            year_data_dict = {}
            for feat_name in self.features_names:
                path = os.path.join(self.data_dir, 'test', f'year_{year_idx}', feat_name + '.npy')
                feat = np.load(path)
                try:
                    next_year_path = os.path.join(self.data_dir, 'test', f'year_{year_idx+1}', feat_name + '.npy')
                    next_year_feat = np.load(next_year_path)
                    feat = np.concatenate([feat, next_year_feat], axis=0)
                except FileNotFoundError:
                    print(f'Cannot find the next year data')
                    print('The initial time will be adjusted to make sure '
                          'lead time will not be extended to the next year')

                if len(feat.shape) == 4:  # t, levels, nlon, nlat -> t, nlat, nlon, levels
                    feat = np.transpose(feat, (0, 3, 2, 1))
                elif len(feat.shape) == 3:  # t, nlon, nlat -> t, nlat, nlon
                    feat = np.transpose(feat, (0, 2, 1))
                year_data_dict[feat_name] = feat

            for start_stamp_idx in stamp_of_year:

                end_stamp_idx = start_stamp_idx + self.nsteps*self.interval
                if end_stamp_idx >= year_data_dict[self.features_names[0]].shape[0]:
                    continue
                feat_dict = {k:
                             torch.from_numpy(v[start_stamp_idx:end_stamp_idx:self.interval]).float()
                             for k, v in year_data_dict.items()}
                start_dayofyear = start_stamp_idx // 4
                start_hour = (start_stamp_idx % 4) * 6

                # append time to the feature dict
                feat_dict['time_dayofyear'] = torch.tensor([start_dayofyear])
                feat_dict['time_hour'] = torch.tensor([start_hour])
                feat_dict['time_year'] = torch.tensor([year_idx])

                yield feat_dict


