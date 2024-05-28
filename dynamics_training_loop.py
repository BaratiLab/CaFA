import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from einops import rearrange, repeat, reduce
import os
import logging, pickle
import yaml, shutil

import wandb
from matplotlib import pyplot as plt

from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.training_utils import ensure_dir, save_checkpoint, load_checkpoint, \
                            dict2namespace

from utils.loss_utils import WeightedLoss, latitude_weighted_rmse, apply_loss_fn_to_dict
from utils.training_utils import plot_result_2d
from weather_transformer import CaFAEPD
from dataset.era5_iter import ERA5Reader, WeatherForecastData, ERA5EvalReader
from dataset.era5 import ResidualNormalizer
from torch.utils.data import DataLoader
import torch.distributed as dist

from torch.utils.checkpoint import checkpoint

import copy
# dataparallel
from torch.nn.parallel import DataParallel



def prepare_training(args, config):

    log_dir = config.log_dir

    # prepare the logger
    # ensure the directory to save the model
    # first check if the log directory exists
    if not torch.distributed.is_initialized() or dist.get_rank() == 0:  # real logger

        ensure_dir(log_dir)
        ensure_dir(log_dir + '/model')
        ensure_dir(log_dir + '/code_cache')
        ensure_dir(log_dir + '/images')

        logger = logging.getLogger("LOG")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, 'logging_info'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # copy config yaml file to log_dir
        shutil.copyfile(args.config, os.path.join(log_dir, 'config.yml'))
        # copy all the code to code_cache folder, including current training script
        shutil.copytree('libs/', os.path.join(log_dir, 'code_cache', 'libs'), dirs_exist_ok=True)
        shutil.copytree('dataset/', os.path.join(log_dir, 'code_cache', 'dataset'), dirs_exist_ok=True)
        shutil.copytree('utils/', os.path.join(log_dir, 'code_cache', 'utils'), dirs_exist_ok=True)
        shutil.copyfile('weather_transformer.py', os.path.join(log_dir, 'code_cache', 'weather_transformer.py'))
        shutil.copyfile('dynamics_training_loop.py', os.path.join(log_dir, 'code_cache', 'dynamics_training_loop.py'))
        shutil.copyfile('train_curriculum_EPD_ddp.py', os.path.join(log_dir, 'code_cache', 'train_curriculum_EPD_ddp.py'))

    else:   # dummy logger (does nothing)
        logger = logging.getLogger("LOG")
        logger.addHandler(logging.NullHandler())

    return logger, log_dir


def configure_epd_models(config):
    model = CaFAEPD(config)
    return model


def configure_optimizers(config, model):
    decay = []
    no_decay = []
    for name, m in model.named_parameters():
        if "spherical_pe" or 'Basis' in name:
            no_decay.append(m)
        else:
            decay.append(m)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay,
                "lr": config.training.lr,
                "betas": (config.training.beta_1, config.training.beta_2),
                "weight_decay": config.training.weight_decay,
            },
            {
                "params": no_decay,
                "lr": config.training.lr,
                "betas": (config.training.beta_1, config.training.beta_2),
                "weight_decay": 0,
            },
        ]
    )

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        config.training.warmup_steps,
        config.training.max_steps,
        config.training.warmup_start_lr,
        config.training.eta_min,
    )

    return optimizer, scheduler


def configure_residual_stat(config):
    # open pre-computed residual scaling factor
    variable_groups = config.data.variable_groups
    feature_names = [item for sublist in variable_groups for item in sublist]
    variable_levels = config.data.variable_levels
    surface_residual_scaling = []
    multi_level_residual_scaling = []

    surface_residual_bias = []
    multi_level_residual_bias = []

    with np.load(os.path.join(config.data.data_dir, 'residual_norm_stats.npz'), allow_pickle=True) as f:
        data_std = f['residual_std'].item()

        for i, feat_name in enumerate(feature_names):
            if variable_levels[i] == 13:
                multi_level_residual_scaling.append(data_std[feat_name].reshape(-1, 1))
                multi_level_residual_bias.append(data_std[feat_name].reshape(-1, 1))
            else:
                surface_residual_scaling.append(data_std[feat_name])
                surface_residual_bias.append(data_std[feat_name])
    surface_residual_scaling = np.array(surface_residual_scaling)  #  [c]
    multi_level_residual_scaling = np.concatenate(multi_level_residual_scaling, axis=-1)  # [nlevels, c]
    surface_residual_scaling = torch.tensor(surface_residual_scaling).float()
    multi_level_residual_scaling = torch.tensor(multi_level_residual_scaling).float()

    surface_residual_bias = np.array(surface_residual_bias)  #  [c]
    multi_level_residual_bias = np.concatenate(multi_level_residual_bias, axis=-1)  # [nlevels, c]
    surface_residual_bias = torch.tensor(surface_residual_bias).float()
    multi_level_residual_bias = torch.tensor(multi_level_residual_bias).float()

    residual_normalizer = ResidualNormalizer(surface_residual_bias, surface_residual_scaling,
                                                multi_level_residual_bias, multi_level_residual_scaling)
    return residual_normalizer


    # if dist.get_rank() == 0:
    #     print('Surface residual scaling:', surface_residual_scaling.shape)
    #     print('Multi-level residual scaling:', multi_level_residual_scaling.shape)



def configure_val_dataset(valsteps, config):
    data_dir = config.data.data_dir
    interval = config.data.interval
    variable_groups = config.data.variable_groups
    # flatten the variable groups
    feature_names = [item for sublist in variable_groups for item in sublist]
    constant_names = config.data.constant_names
    variable_levels = config.data.variable_levels

    val_dataset = WeatherForecastData(
        ERA5Reader(data_dir, feature_names, constant_names, variable_levels,
                   'valid', '0/12', interval, valsteps))

    return val_dataset


def configure_train_dataset_and_loader(trainsteps, batch_size, config):
    data_dir = config.data.data_dir
    interval = config.data.interval
    variable_groups = config.data.variable_groups
    # flatten the variable groups
    feature_names = [item for sublist in variable_groups for item in sublist]
    constant_names = config.data.constant_names
    variable_levels = config.data.variable_levels
    # get tining year range
    years_range = config.data.years_range

    train_dataset = WeatherForecastData(
        ERA5Reader(data_dir, feature_names, constant_names, variable_levels,
                   'train', 'all', interval, trainsteps,
                   years_range=years_range,
                   shuffle=True))
    if dist.is_initialized():
        train_dataloader = DataLoader(train_dataset,
                                    batch_size=int(batch_size // dist.get_world_size()),
                                    num_workers=config.training.train_num_workers,
                                    shuffle=False,
                                    # sampler=sampler,
                                    pin_memory=True, drop_last=True)
    else:
        train_dataloader = DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    num_workers=config.training.train_num_workers,
                                    shuffle=False,
                                    pin_memory=True, drop_last=True)

    return train_dataset, train_dataloader


def configure_test_dataset(config, teststeps):
    data_dir = config.data.data_dir
    interval = config.data.interval
    variable_groups = config.data.variable_groups
    # flatten the variable groups
    feature_names = [item for sublist in variable_groups for item in sublist]
    constant_names = config.data.constant_names
    variable_levels = config.data.variable_levels
    start_time_limit_years = config.data.start_time_limit_years
    init_time = config.testing.init_time

    val_dataset = WeatherForecastData(
        ERA5EvalReader(data_dir, feature_names, constant_names, variable_levels,
                       init_time, interval, teststeps, start_time_limit_years=start_time_limit_years))

    return val_dataset


def configure_loss(config):
    loss_module = WeightedLoss(loss_fn=nn.SmoothL1Loss(beta=config.training.smooth_l1_beta, reduction='none'),
                               latitude_resolution=config.data.nlat,
                               level_weight=config.training.level_weight,
                               multi_level_variable_weight=config.data.multi_level_variable_weight,
                               surface_variable_weight=config.data.surface_variable_weight)
    return loss_module


def dump_state(model, optimizer, scheduler, global_step, log_dir, ema=False):
    if not ema:
        state_dict = {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'global_step': global_step,
        }
        save_checkpoint(state_dict, log_dir + '/model' + f'/model_{(global_step // 1000)}k_iter.pth')

    else:
        state_dict = {
            'model': model.state_dict(),   # this is a ema model
        }
        save_checkpoint(state_dict, log_dir + '/model' + f'/ema_{(global_step // 1000)}k_iter.pth')


def load_state(model, checkpoint, config):
    model.load_state_dict(checkpoint['model'])
    optim, sched = configure_optimizers(config, model)
    optim.load_state_dict(checkpoint['optimizer'])
    sched.load_state_dict(checkpoint['scheduler'])
    global_step = checkpoint['global_step']
    return optim, sched, global_step


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def epd_train_step(model,
               surface_in_feat,
               surface_target_feat,
               multi_level_in_feat,
               multi_level_target_feat,
               constants,
               optimizer,
               scheduler,
               loss_module,
               grad_post_fn,
               residual_normalizer
               ):

    model.train()
    optimizer.zero_grad()

    # train the autoencoder + processor for dynamics prediction

    out_T = surface_target_feat.shape[1]
    pred_surface_feat = torch.zeros_like(surface_target_feat)
    pred_multi_level_feat = torch.zeros_like(multi_level_target_feat)

    for step in range(out_T):
        pred_surface_residual_t, pred_multi_level_residual_t = model(surface_in_feat,
                                                             multi_level_in_feat,
                                                             constants)
        pred_surface_residual_t, pred_multi_level_residual_t = \
            residual_normalizer.scale_and_offset(pred_surface_residual_t, pred_multi_level_residual_t)

        pred_surface_feat_t = surface_in_feat + pred_surface_residual_t
        pred_multi_level_feat_t = multi_level_in_feat + pred_multi_level_residual_t

        pred_surface_feat[:, step] = pred_surface_feat_t
        pred_multi_level_feat[:, step] = pred_multi_level_feat_t

        surface_in_feat = pred_surface_feat_t
        multi_level_in_feat = pred_multi_level_feat_t

    loss = loss_module(pred_surface_feat, surface_target_feat, pred_multi_level_feat, multi_level_target_feat)
    loss.backward()
    # clip gradients
    grad_post_fn(model)
    optimizer.step()
    scheduler.step()
    return loss.detach().item()


def epd_train_step_with_checkpoint(model,
               surface_in_feat,
               surface_target_feat,
               multi_level_in_feat,
               multi_level_target_feat,
               constants,
               optimizer,
               scheduler,
               loss_module,
               grad_post_fn,
               residual_normalizer,
               segment_size,   # for gradient checkpointing
               ):

    model.train()
    optimizer.zero_grad()
    out_T = surface_target_feat.shape[1]    # total rollout steps
    assert out_T > segment_size, 'segment size should be smaller than the total rollout steps'

    # train the autoencoder + processor for dynamics prediction
    # define custom forward
    def run_function(start, end, constants):
        outputs = []
        def custom_forward(*inputs):
            for step in range(start, end):
                pred_surface_residual, pred_multi_level_residual = model(inputs[0],
                                                                     inputs[1],
                                                                     constants)
                pred_surface_residual_t, pred_multi_level_residual_t = \
                    residual_normalizer.scale_and_offset(pred_surface_residual, pred_multi_level_residual)

                pred_surface_feat_t = inputs[0] + pred_surface_residual_t
                pred_multi_level_feat_t = inputs[1] + pred_multi_level_residual_t
                outputs.append((pred_surface_feat_t, pred_multi_level_feat_t))
                inputs = (pred_surface_feat_t, pred_multi_level_feat_t)
            return outputs
        return custom_forward

    segments = out_T // segment_size
    if out_T % segment_size > 0:
        segments += 1

    pred_lst = []
    for i in range(segments):
        start = i * segment_size
        end = min((i + 1) * segment_size, out_T)
        outputs = checkpoint(run_function(start, end, constants), surface_in_feat, multi_level_in_feat, use_reentrant=False)
        pred_lst.extend(outputs)
        surface_in_feat = outputs[-1][0]
        multi_level_in_feat = outputs[-1][1]

    pred_surface_feat = torch.stack([x[0] for x in pred_lst], dim=1)
    pred_multi_level_feat = torch.stack([x[1] for x in pred_lst], dim=1)
    loss = loss_module(pred_surface_feat, surface_target_feat, pred_multi_level_feat, multi_level_target_feat)
    loss.backward()
    # clip gradients
    grad_post_fn(model)
    optimizer.step()
    scheduler.step()
    return loss.detach().item()


def epd_predict(model,
                timestamps,
                in_feat_dict,
                out_feat_dict,
                constants,
                normalizer,
                residual_normalizer,
                loss_fn,
                config,
                device,
                return_pred=False):
    model.eval()

    # to device
    surface_in_feat = []
    multi_level_in_feat = []

    # prepare input into surface and multi-level features
    variable_levels = config.data.variable_levels
    normalizer.normalize(in_feat_dict)
    for i, key in enumerate(in_feat_dict.keys()):
        v = in_feat_dict[key]
        if variable_levels[i] == 1:
            surface_in_feat.append(v)
        elif variable_levels[i] > 1:
            multi_level_in_feat.append(v)

    surface_in_feat = torch.stack(surface_in_feat, dim=-1).to(device)
    multi_level_in_feat = torch.stack(multi_level_in_feat, dim=-1).to(device)
    constants = constants.to(device)

    out_T = out_feat_dict[list(out_feat_dict.keys())[0]].shape[1]
    batch_size, nlat, nlon, c_surface = surface_in_feat.shape
    batch_size, nlat, nlon, nlevels, c_multi_level = multi_level_in_feat.shape

    pred_surface_feat = torch.zeros((batch_size, len(timestamps), nlat, nlon, c_surface)).to(device)
    pred_multi_level_feat = torch.zeros((batch_size, len(timestamps), nlat, nlon, nlevels, c_multi_level)).to(device)
    max_timestamp = max(timestamps)
    with torch.inference_mode():
        for step in range(max_timestamp):
            pred_surface_residual_t, pred_multi_level_residual_t = model(surface_in_feat,
                                                                 multi_level_in_feat,
                                                                 constants)
            pred_surface_residual_t, pred_multi_level_residual_t = \
                residual_normalizer.scale_and_offset(pred_surface_residual_t, pred_multi_level_residual_t)

            pred_surface_feat_t = surface_in_feat + pred_surface_residual_t
            pred_multi_level_feat_t = multi_level_in_feat + pred_multi_level_residual_t
            if step+1 in timestamps:
                pred_surface_feat[:, timestamps.index(step+1)] = pred_surface_feat_t
                pred_multi_level_feat[:, timestamps.index(step+1)] = pred_multi_level_feat_t

            surface_in_feat = pred_surface_feat_t
            multi_level_in_feat = pred_multi_level_feat_t

    pred_feat_dict = {}
    c1 = 0  # index for surface features
    c2 = 0  # index for multi-level features
    for i, k in enumerate(out_feat_dict.keys()):
        out_feat_dict[k] = out_feat_dict[k].index_select(1, torch.tensor(timestamps) - 1).to(device)

        # we dont need the reconstruction results
        if variable_levels[i] == 1:
            pred_feat_dict[k] = pred_surface_feat[..., c1]

            c1 += 1
        elif variable_levels[i] > 1:
            pred_feat_dict[k] = pred_multi_level_feat[..., c2]
            c2 += 1


    normalizer.batch_denormalize(pred_feat_dict)
    if not return_pred:
        return apply_loss_fn_to_dict(pred_feat_dict, out_feat_dict, loss_fn)
    else:
        return apply_loss_fn_to_dict(pred_feat_dict, out_feat_dict, loss_fn), pred_feat_dict


@torch.no_grad()
def validate_loop(model,
                  timestamps,
                  logger,
                  global_step,
                  val_dataset,
                  val_batch_size,
                  config,
                  device):
    print('Validating...')
    logger.info('====================================')
    logger.info('Validating...')
    logger.info(f'Iter steps: {global_step}')
    model = copy.deepcopy(model)
    model.eval()
    BS = val_batch_size
    val_dataloader = DataLoader(val_dataset,
                                batch_size=BS)
    val_loss_fn = latitude_weighted_rmse
    val_loss_dict = {}
    # randomly select a batch
    i_vis = 3
    pbar = tqdm(val_dataloader)
    residual_normalizer = configure_residual_stat(config)
    residual_normalizer = residual_normalizer.to(device)
    for i, batch in enumerate(pbar):
        input_dict, target_dict, constants, _, _, _ = batch
        loss_dict, pred_dict = epd_predict(model, timestamps, input_dict, target_dict, constants,
                                               val_dataset.datareader.normalizer,
                                               residual_normalizer,
                                               val_loss_fn,
                                               config=config,
                                               device=device,
                                               return_pred=True)
        for k in loss_dict.keys():
            if k not in val_loss_dict.keys():
                val_loss_dict[k] = []
            val_loss_dict[k].append(loss_dict[k].cpu().detach().numpy())

        # hard coded to visualize the 2m temperature and geopotential 500hPa
        if i == i_vis:
            target = target_dict['2m_temperature'].cpu().detach().numpy()
            pred = pred_dict['2m_temperature'].cpu().detach().numpy()
            plot_result_2d(target, pred,
                           num_t=target.shape[1],
                           filename=os.path.join(config.log_dir, 'images',
                                                 f'val_{i_vis}_iter_t2m:{global_step}.png'))

            target = target_dict['geopotential'].cpu().detach().numpy()
            pred = pred_dict['geopotential'].cpu().detach().numpy()
            plot_result_2d(target[..., 6], pred[..., 6],
                            num_t=target.shape[1],
                            filename=os.path.join(config.log_dir, 'images',
                                                     f'val_{i_vis}_iter_z500:{global_step}.png'))

    for k in val_loss_dict.keys():
        v = np.concatenate(val_loss_dict[k], axis=0)  # stack along batch
        v = np.mean(v, axis=0)  # [time, level] or [time]
        interval = config.data.interval

        idx_72, idx_120, idx_168 = \
            timestamps.index(int(72 / (interval * 6)) ), timestamps.index(int(120 / (interval * 6)) ), \
            timestamps.index(int(168 / (interval * 6)) )

        if 'temperature' in k or 'geopotential' in k:
            if len(v.shape) == 2:
                for l_num in range(v.shape[1]):
                    if ((l_num == v.shape[1] - 3 and k =='temperature') or  # 850hPa
                            (l_num == v.shape[1] - 6 and k == 'geopotential')):  # 500hPa
                        print(f'Validation rmse for {k}_{l_num} at 72hr/120hr/240hr:'
                              f'{v[idx_72, l_num]:.4f}/{v[idx_120, l_num]:.4f}/{v[idx_168, l_num]:.4f}')
                    logger.info(f'Validation rmse for {k}_{l_num} at 72hr/120hr/168hr:'
                                    f'{v[idx_72, l_num]:.4f}/{v[idx_120, l_num]:.4f}/{v[idx_168, l_num]:.4f}')
                    wandb.log({
                        f'val_rmse_{k}_{l_num}_next': np.round(v[0, l_num], 4),
                        f'val_rmse_{k}_{l_num}_72hr': np.round(v[idx_72, l_num], 4),
                        f'val_rmse_{k}_{l_num}_120hr': np.round(v[idx_120, l_num], 4),
                        f'val_rmse_{k}_{l_num}_168hr': np.round(v[idx_168, l_num], 4),
                    })
            elif len(v.shape) == 1:
                print(f'Validation rmse for {k} at 72hr/120hr/168hr:'
                      f'{v[idx_72]:.4f}/{v[idx_120]:.4f}/{v[idx_168]:.4f}')

                wandb.log({
                    f'val_rmse_{k}_next': np.round(v[0], 4),
                    f'val_rmse_{k}_72hr': np.round(v[idx_72], 4),
                    f'val_rmse_{k}_120hr': np.round(v[idx_120], 4),
                    f'val_rmse_{k}_168hr': np.round(v[idx_168], 4),
                })
            else:
                raise ValueError('Invalid shape of v')
    logger.info('====================================')
    # clear cuda cache
    del model
    torch.cuda.empty_cache()
    return


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--comment', type=str, default='', help='Comment')
    parser.add_argument('--global_seed', type=int, default=970314, help='Global seed')
    args = parser.parse_args()

    # parse config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    # copy the config file to the log_dir
    return args, config



