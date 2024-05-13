import numpy as np
import torch
import torch.nn as nn
import xarray as xr
import dataclasses


def get_cosine_weight(num_intervals, tau):
    start = 0
    end = 1
    t = np.linspace(0, 1, num_intervals+1)
    v_start = np.cos(start * np.pi / 2) ** (2 * tau)
    v_end = np.cos(end * np.pi / 2) ** (2 * tau)
    output = np.cos((t * (end - start) + start) * np.pi / 2) ** (2 * tau)
    output = 1 - (v_end - output) / (v_end - v_start)
    return output[1:]


# base on the code from graphcast
def _check_uniform_spacing_and_get_delta(vector):
    diff = np.diff(vector)
    if not np.all(np.isclose(diff[0], diff)):
        raise ValueError(f'Vector {diff} is not uniformly spaced.')
    return diff[0]


def _weight_for_latitude_vector_without_poles(latitude):
    """Weights for uniform latitudes of the form [+-90-+d/2, ..., -+90+-d/2]."""
    delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
    if (not np.isclose(np.max(latitude), 90 - delta_latitude/2) or
        not np.isclose(np.min(latitude), -90 + delta_latitude/2)):
        raise ValueError(
            f'Latitude vector {latitude} does not start/end at '
            '+- (90 - delta_latitude/2) degrees.')
    return np.cos(np.deg2rad(latitude))


def _weight_for_latitude_vector_with_poles(latitude):
    """Weights for uniform latitudes of the form [+- 90, ..., -+90]."""
    delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
    if (not np.isclose(np.max(latitude), 90.) or
        not np.isclose(np.min(latitude), -90.)):
        raise ValueError(
            f'Latitude vector {latitude} does not start/end at +- 90 degrees.')
    weights = np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(delta_latitude/2))
    # The two checks above enough to guarantee that latitudes are sorted, so
    # the extremes are the poles
    weights[[0, -1]] = np.sin(np.deg2rad(delta_latitude/4)) ** 2
    return weights


class WeightedLoss(nn.Module):
    def __init__(self,
                 loss_fn,
                 latitude_resolution,
                 latitude_weight='cosine',
                 level_weight='linear',
                 multi_level_variable_weight=None,
                 surface_variable_weight=None,
                 ):
        super().__init__()
        self.loss_fn = loss_fn   # loss function must not reduce any dimension
        if latitude_weight == 'cosine':
            latitude = np.linspace(-90, 90, latitude_resolution)   # currently we use dataset that contains poles
            weights = torch.from_numpy(_weight_for_latitude_vector_with_poles(latitude))
            latitude_weight = weights / weights.mean()
        else:
            weights = torch.ones(latitude_resolution)   # all latitudes weight the same
            latitude_weight = weights / weights.mean()
        self.register_buffer('latitude_weight', latitude_weight)

        # weight for each level
        # up to 13 pressure levels, the lower the level, the lower the weight
        # 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000
        if level_weight == 'linear':     # slightly outweighs the lower levels
            level_weight = torch.linspace(0.05, 0.065, 13)
        elif level_weight == 'exp':
            level_weight = torch.exp(torch.linspace(-3, 0, 13))
            level_weight = level_weight / level_weight.sum()
        elif level_weight == 'cosine':
            level_weight = torch.from_numpy(get_cosine_weight(13, 2))
            level_weight = level_weight / level_weight.sum()
        else:
            level_weight = torch.ones(13)
            level_weight = level_weight / level_weight.sum()
        self.register_buffer('level_weight', level_weight)

        if surface_variable_weight is not None:
            surface_variable_weight = torch.tensor(surface_variable_weight)
        else:
            surface_variable_weight = torch.tensor(1.)
        self.register_buffer('surface_variable_weight', surface_variable_weight)

        if multi_level_variable_weight is not None:
            multi_level_variable_weight = torch.tensor(multi_level_variable_weight)
        else:
            multi_level_variable_weight = torch.tensor(1.)
        self.register_buffer('multi_level_variable_weight', multi_level_variable_weight)

    def forward(self,
                surface_pred_feat,      # [b, t, nlat, nlon, c]
                surface_target_feat,
                multi_level_pred_feat,   # [b, t, nlat, nlon, l, c]
                multi_level_target_feat,
                ):
        nlat = multi_level_target_feat.shape[2]

        latitude_weight = self.latitude_weight.view(1, 1, -1, 1)
        surface_loss = self.loss_fn(surface_pred_feat, surface_target_feat) * self.surface_variable_weight
        surface_loss = surface_loss.sum(dim=-1)
        multi_level_loss = self.loss_fn(multi_level_pred_feat, multi_level_target_feat) * self.level_weight.view(1, 1,
                                                                                                                 1, 1,
                                                                                                                 -1, 1)
        multi_level_loss = (multi_level_loss.sum(dim=-2) * self.multi_level_variable_weight).sum(dim=-1)

        loss = surface_loss + multi_level_loss
        loss = loss * latitude_weight

        return loss.mean()   # reduce over batch/time/lat/lon


def latitude_weighted_rmse(pred, target):
    # directly infer latitude from target: b t nlat nlon or b t nlat nlon l
    nlat = target.shape[2]
    lat_weight = _weight_for_latitude_vector_with_poles(np.linspace(-90, 90, nlat))
    lat_weight = torch.from_numpy(lat_weight).to(target.device)
    lat_weight = lat_weight / lat_weight.mean()
    if len(pred.shape) == 5:
        lat_weight = lat_weight.view(1, 1, nlat, 1, 1)
    else:
        lat_weight = lat_weight.view(1, 1, nlat, 1)

    return torch.sqrt((((pred - target)**2) * lat_weight).mean(dim=(2, 3)))   # spatial averaging


def latitude_weighted_l1(pred, target):
    # directly infer latitude from target: b t nlat nlon or b t nlat nlon l
    nlat = target.shape[2]
    lat_weight = _weight_for_latitude_vector_with_poles(np.linspace(-90, 90, nlat))
    lat_weight = torch.from_numpy(lat_weight).to(target.device)
    lat_weight = lat_weight / lat_weight.mean()
    if len(pred.shape) == 5:
        lat_weight = lat_weight.view(1, 1, nlat, 1, 1)
    else:
        lat_weight = lat_weight.view(1, 1, nlat, 1)

    return ((pred - target).abs() * lat_weight).mean(dim=(2, 3))   # spatial averaging


def apply_loss_fn_to_dict(pred_dict, target_dict, loss_fn, **kwargs):
    loss_dict = {}
    for k in pred_dict.keys():
        loss_dict[k] = loss_fn(pred_dict[k], target_dict[k], **kwargs)

    return loss_dict


# below are some re-cast of Weatherbench2's code:
# https://github.com/google-research/weatherbench2/blob/2aa282a6dca3c88f1941aea341a02ba3d81358aa/weatherbench2/metrics.py
def _get_climatology_chunk(
    climatology: xr.Dataset, key   # python dict
    ) -> xr.Dataset:
  """Returns the climatological mean of the observed true variables."""
  try:
    climatology_chunk = climatology[key]
  except KeyError as e:
    not_found = set(key).difference(climatology.data_vars)
    clim_var = key + "_mean"  # pytype: disable=unsupported-operands
    not_found_means = set(clim_var).difference(climatology.data_vars)
    if not_found and not_found_means:
      raise KeyError(
          f"Did not find {not_found} keys in climatology. Appending "
          "'mean' did not help."
      ) from e
    climatology_chunk = climatology[key].rename(
        clim_var
    )
  return climatology_chunk


def _spatial_average(x, weight='lat'):
    # assuming that x must in shape: [b, t, nlat, nlon] or [b, t, nlat, nlon, l]
    # by default, we use latitude weight
    if weight == 'lat':
        nlat = x.shape[2]
        lat_weight = _weight_for_latitude_vector_with_poles(np.linspace(-90, 90, nlat))
        lat_weight = torch.from_numpy(lat_weight).to(x.device)
        lat_weight = lat_weight / lat_weight.mean()
        if len(x.shape) == 5:
            lat_weight = lat_weight.view(1, 1, nlat, 1, 1)
        else:
            lat_weight = lat_weight.view(1, 1, nlat, 1)
        return (x * lat_weight).mean(dim=(2, 3))
    else:
        return x.mean(dim=(2, 3))



@dataclasses.dataclass
class ACC:
    """Anomaly correlation coefficient.

    Attribute:
    climatology: Climatology for computing anomalies.
    """

    climatology: xr.Dataset

    def compute(self,
                pred,
                target,   # [b, t, nlat, nlon] or [b, t, nlat, nlon, l]
                data_key,
                hour_start,
                day_start,
                interval):

        climatology_chunk = _get_climatology_chunk(self.climatology, data_key)

        # loop through batches
        climatology_chunk_batched = torch.zeros(pred.shape)
        for b_idx in range(target.shape[0]):

            b_hour_start = int(hour_start[b_idx])
            b_day_start = int(day_start[b_idx])

            # compute time_end assuming dt is 6 hours
            nt = target.shape[1]
            dt = 6 * interval
            hrofday = b_hour_start + (np.arange(nt)+1) * dt
            hrofday = hrofday % 24

            # compute dayofyear, original day_start is 0-based
            dayofyear = b_day_start + 1 + (b_hour_start + dt * (np.arange(nt)+1)) // 24

            # wrap dayofyear into 366, all the dayof year will be ranging in [1, 366]
            dayofyear[dayofyear >= 367] = (dayofyear[dayofyear >= 367] % 367 + 1)

            try:
                c = climatology_chunk.sel(dayofyear=xr.DataArray(dayofyear),
                                          hour=xr.DataArray(hrofday)).to_numpy()
            except KeyError:
                print('Cannot find the following dayofyear and hour:')
                print('dayofyear:', dayofyear)
                print('hour:', hrofday)
                exit()

            if len(c.shape) == 3:
                c = np.transpose(c, (0, 2, 1))
            else:
                c = np.transpose(c, (0, 3, 2, 1))
            climatology_chunk_batched[b_idx] = \
                torch.from_numpy(c)

        climatology_chunk_batched = climatology_chunk_batched.to(pred.device)

        forecast_anom = pred - climatology_chunk_batched
        truth_anom = target - climatology_chunk_batched
        return _spatial_average(
            forecast_anom * truth_anom,
        ) / torch.sqrt(
            _spatial_average(forecast_anom ** 2)
            * _spatial_average(truth_anom ** 2)
        )