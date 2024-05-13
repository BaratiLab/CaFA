import os
import argparse
import shutil
import torch
import matplotlib
from matplotlib import pyplot as plt
import numpy as np


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def save_checkpoint(state, save_path: str, is_best: bool = False, max_keep: int = None):
    """Saves torch model to checkpoint file.
    Args:
        state (torch model state): State of a torch Neural Network
        save_path (str): Destination path for saving checkpoint
        is_best (bool): If ``True`` creates additional copy
            ``best_model.ckpt``
        max_keep (int): Specifies the max amount of checkpoints to keep
    """
    # save checkpoint
    torch.save(state, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint.txt')

    save_path = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_path + '\n'] + ckpt_list
    else:
        ckpt_list = [save_path + '\n']

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))


def load_checkpoint(ckpt_dir_or_file: str, map_location=None, load_best=False):
    """Loads torch model from checkpoint file.
    Args:
        ckpt_dir_or_file (str): Path to checkpoint directory or filename
        map_location: Can be used to directly load to specific device
        load_best (bool): If True loads ``best_model.ckpt`` if exists.
    """
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
        else:
            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint.txt')) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


def ensure_dir(dir_name: str):
    """Creates folder if not exists.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def plot_error(err, title, dt=0.25, metric='Latitude weighted RMSE'):
    # error should be in shape [timesteps]
    # dt in the unit of days
    fig, ax = plt.subplots(figsize=[6, 4], dpi=200)

    # plot error
    ax.plot(np.arange(1, len(err)+1)*dt, err, color='b')

    # Hide the right and top spines
    plt.ylabel(f'{metric}', fontsize=12)
    plt.xlabel('Time (days)', fontsize=12)
    plt.grid(which='both', linestyle='-.')

    plt.title(title, fontsize=12)
    return fig


from mpl_toolkits.axes_grid1 import ImageGrid


def plot_result_2d(y, y_pred, filename,
                   num_vis=3, num_t=6, cmap='RdBu'):
    matplotlib.use('Agg')

    # visualize the result in 2D rectangular map
    # y and y_pred should be in shape [Nsample, time, lat, lon]
    # we randomly pick num_vis samples to visualize
    # num_vis: number of samples to visualize

    # the visualization are arranged as follows:
    # first row: y[0, 0, :, :], y[0, t, :, :], y[0, 2*t, :, :],..., y[0, T, :, :]
    # second row: y_pred[0, 0, :, :], y_pred[0, t, :, :], y_pred[0, 2*t, :, :],..., y_pred[0, T, :, :]
    # third row: y[1, 0, :, :], y[1, t, :, :], y[1, 2*t, :, :],..., y[1, T, :, :] and so on

    _, t_total, h, w = y_pred.shape
    dt = t_total // num_t
    fig = plt.figure(figsize=(12, 6))

    y_pred = y_pred[:num_vis, ::dt, :, :]
    y = y[:num_vis, ::dt, :, :]

    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(num_vis*2, num_t),
                     axes_pad=0.05,
                     share_all=False,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="3%",
                     cbar_pad=0.15,
                     )

    # Add data to image grid
    for row in range(num_vis):
        for t in range(num_t):
            grid[row*2*num_t + t].imshow(y_pred[row, t], cmap=cmap)
            grid[row*2*num_t + t].axis('off')
            im = grid[row*2*num_t + t + num_t].imshow(y[row, t], cmap=cmap)
            grid[row*2*num_t + t + num_t].axis('off')
            grid[row*2*num_t + t + num_t].cax.colorbar(im)
            grid[row*2*num_t + t + num_t].cax.toggle_label(True)

    # save the figure
    plt.savefig(filename, dpi=200)
    plt.close()


@torch.no_grad()
def ema_update(ema_model, model, decay=0.999, copy_buffer=False):
    """Update the EMA model with the current model parameters.
    Args:
        ema_model (torch.nn.Module): The EMA model
        model (torch.nn.Module): The current model
        decay (float): The decay rate
    """
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.copy_(param.detach().lerp(ema_param, decay))

    # copy the buffers
    if copy_buffer:
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)



