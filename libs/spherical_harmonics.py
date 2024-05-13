from math import pi, sqrt
from functools import reduce
from operator import mul
import torch

import os
import sys
import time
import pickle
import gzip
import torch
import contextlib
from functools import wraps, lru_cache

# caching functions

def exists(val):
    return val is not None


def cache(cache, key_fn):
    def cache_inner(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            key_name = key_fn(*args, **kwargs)
            if key_name in cache:
                return cache[key_name]
            res = fn(*args, **kwargs)
            cache[key_name] = res
            return res

        return inner
    return cache_inner

# cache in directory
def cache_dir(dirname, maxsize=128):
    '''
    Cache a function with a directory

    :param dirname: the directory path
    :param maxsize: maximum size of the RAM cache (there is no limit for the directory cache)
    '''
    def decorator(func):

        @lru_cache(maxsize=maxsize)
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not exists(dirname):
                return func(*args, **kwargs)

            os.makedirs(dirname, exist_ok = True)

            indexfile = os.path.join(dirname, "index.pkl")
            lock = FileLock(os.path.join(dirname, "mutex"))

            with lock:
                index = {}
                if os.path.exists(indexfile):
                    with open(indexfile, "rb") as file:
                        index = pickle.load(file)

                key = (args, frozenset(kwargs), func.__defaults__)

                if key in index:
                    filename = index[key]
                else:
                    index[key] = filename = f"{len(index)}.pkl.gz"
                    with open(indexfile, "wb") as file:
                        pickle.dump(index, file)

            filepath = os.path.join(dirname, filename)

            if os.path.exists(filepath):
                with lock:
                    with gzip.open(filepath, "rb") as file:
                        result = pickle.load(file)
                return result

            print(f"compute {filename}... ", end="", flush = True)
            result = func(*args, **kwargs)
            print(f"save {filename}... ", end="", flush = True)

            with lock:
                with gzip.open(filepath, "wb") as file:
                    pickle.dump(result, file)

            print("done")

            return result
        return wrapper
    return decorator


# constants

CACHE = {}


def clear_spherical_harmonics_cache():
    CACHE.clear()


def lpmv_cache_key_fn(l, m, x):
    return (l, m)


# spherical harmonics

@lru_cache(maxsize=1000)
def semifactorial(x):
    return reduce(mul, range(x, 1, -2), 1.)


@lru_cache(maxsize=1000)
def pochhammer(x, k):
    return reduce(mul, range(x + 1, x + k), float(x))


def negative_lpmv(l, m, y):
    if m < 0:
        y *= ((-1) ** m / pochhammer(l + m + 1, -2 * m))
    return y


@cache(cache=CACHE, key_fn=lpmv_cache_key_fn)
def lpmv(l, m, x):
    """Associated Legendre function including Condon-Shortley phase.

    Args:
        m: int order
        l: int degree
        x: float argument tensor
    Returns:
        tensor of x-shape
    """
    # Check memoized versions
    m_abs = abs(m)

    if m_abs > l:
        return None

    if l == 0:
        return torch.ones_like(x)

    # Check if on boundary else recurse solution down to boundary
    if m_abs == l:
        # Compute P_m^m
        y = (-1) ** m_abs * semifactorial(2 * m_abs - 1)
        y *= torch.pow(1 - x * x, m_abs / 2)
        return negative_lpmv(l, m, y)

    # Recursively precompute lower degree harmonics
    lpmv(l - 1, m, x)

    # Compute P_{l}^m from recursion in P_{l-1}^m and P_{l-2}^m
    # Inplace speedup
    y = ((2 * l - 1) / (l - m_abs)) * x * lpmv(l - 1, m_abs, x)

    if l - m_abs > 1:
        y -= ((l + m_abs - 1) / (l - m_abs)) * CACHE[(l - 2, m_abs)]

    if m < 0:
        y = negative_lpmv(l, m, y)
    return y


def get_spherical_harmonics_element(l, m, theta, phi):
    """Tesseral spherical harmonic with Condon-Shortley phase.

    The Tesseral spherical harmonics are also known as the real spherical
    harmonics.

    Args:
        l: int for degree
        m: int for order, where -l <= m < l
        theta: collatitude or polar angle    [nlat,]
        phi: longitude or azimuth            [nlon,]
    Returns:
        tensor of shape theta
    """
    m_abs = abs(m)
    assert m_abs <= l, "absolute value of order m must be <= degree l"

    N = sqrt((2 * l + 1) / (4 * pi))
    leg = lpmv(l, m_abs, torch.cos(theta))  # [nlat,]

    if m == 0:
        return N * leg.unsqueeze(1).expand(-1, phi.shape[0])

    if m > 0:
        Y = torch.cos(m * phi)
    else:
        Y = torch.sin(m_abs * phi)

    Y = Y.unsqueeze(0)
    leg = leg.unsqueeze(1)
    Y = Y*leg
    N *= sqrt(2. / pochhammer(l - m_abs + 1, 2 * m_abs))
    Y *= N
    return Y

@torch.no_grad()
def get_spherical_harmonics(l, theta, phi):
    """ Tesseral harmonic with Condon-Shortley phase.

    The Tesseral spherical harmonics are also known as the real spherical
    harmonics.

    Args:
        l: int for degree
        theta: collatitude or polar angle
        phi: longitude or azimuth
    Returns:
        tensor of shape [*theta.shape, 2*l+1]
    """
    return torch.stack([get_spherical_harmonics_element(l, m, theta, phi) \
                        for m in range(-l, l + 1)],
                       dim=-1)


class SphericalHarmonicsPE(torch.nn.Module):
    def __init__(self, l_max, dim, out_dim, use_mlp=True):
        super().__init__()
        self.l_max = l_max
        self.dim = dim
        self.out_dim = out_dim

        # each l has 2l+1 basis, in total 1 + 3 + 5 + ... + (2l_max+1) = (l_max+1)^2
        if use_mlp:
            basis_weight = torch.randn((self.l_max + 1) ** 2, self.dim) / (self.l_max + 1)
            self.basis_weight = torch.nn.Parameter(basis_weight, requires_grad=True)
            self.pe_mlp = torch.nn.Sequential(
                torch.nn.GELU(),
                torch.nn.Linear(self.dim, self.dim),
                torch.nn.GELU(),
                torch.nn.Linear(self.dim, self.out_dim),
            )
        else:
            basis_weight = torch.randn((self.l_max + 1) ** 2, self.out_dim) / (self.l_max + 1)
            self.basis_weight = torch.nn.Parameter(basis_weight, requires_grad=True)
            self.pe_mlp = torch.nn.Identity()

    def sph(self, lat, lon):
        Y = []
        for l in range(self.l_max + 1):
            Y.append(get_spherical_harmonics(l, lat, lon))
        return Y

    def cache_precomputed_sph_harmonics(self, lat, lon):
        # lat: nlat
        # lon: nlon
        # create meshgrid first
       #  lat, lon = torch.meshgrid([lat, lon], indexing='ij')  # (nlat, nlon)

        sph_lst = self.sph(lat, lon)  # lst of (nlat, nlon, 2l+1)
        sph_harmonics = torch.cat(sph_lst, dim=-1)  # (nlat, nlon, (l_max+1)^2)
        self.register_buffer('sph_harmonics', sph_harmonics, persistent=False)
        clear_spherical_harmonics_cache()   # clear cache so that it wont affect other computations

    def forward(self, lat, lon):
        if not hasattr(self, 'sph_harmonics'):
            self.cache_precomputed_sph_harmonics(lat, lon)
        sph_feat = self.sph_harmonics.detach().clone()  # (nlat, nlon, (l_max+1)^2)
        sph_feat = torch.einsum('ijd,dc->ijc', sph_feat, self.basis_weight)  # (nlat, nlon, dim)
        sph_feat = self.pe_mlp(sph_feat)  # (nlat, nlon, dim)
        return sph_feat



if __name__ == '__main__':
    # visualize spherical harmonics
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # import cartopy
    # import cartopy.crs as ccrs

    lat = torch.from_numpy(np.linspace(0, np.pi, 64)).float()
    lon = torch.from_numpy(np.linspace(-np.pi, np.pi - (2*np.pi/128), 128)).float()

    # lat, lon = torch.meshgrid([lat, lon])

    sph_module = SphericalHarmonicsPE(l_max=7, dim=32, out_dim=32)
    sph_feat = sph_module(lat, lon).detach().numpy()
    plt.imshow(sph_feat[:, :, 0],  cmap="RdBu")
    plt.show()

    # plt.imshow(sph_feat[:, :, 16],  cmap="RdBu")
    # plt.show()

    plt.imshow(sph_feat[:, :, -1],  cmap="RdBu")

    plt.show()
    # def plot_sphere(data,
    #                 fig=None,
    #                 cmap="RdBu",
    #                 title=None,
    #                 colorbar=False,
    #                 coastlines=False,
    #                 central_latitude=20,
    #                 central_longitude=20,
    #                 lon=None,
    #                 lat=None,
    #                 **kwargs):
    #     if fig == None:
    #         fig = plt.figure()
    #
    #     nlat = data.shape[-2]
    #     nlon = data.shape[-1]
    #     if lon is None:
    #         lon = np.linspace(0, 2 * np.pi, nlon)
    #     if lat is None:
    #         lat = np.linspace(np.pi / 2., -np.pi / 2., nlat)
    #     Lon, Lat = np.meshgrid(lon, lat)
    #
    #     proj = ccrs.Orthographic(central_longitude=central_longitude, central_latitude=central_latitude)
    #     # proj = ccrs.Mollweide(central_longitude=central_longitude)
    #
    #     ax = fig.add_subplot(projection=proj)
    #     Lon = Lon * 180 / np.pi
    #     Lat = Lat * 180 / np.pi
    #
    #     # contour data over the map.
    #     im = ax.pcolormesh(Lon, Lat, data, cmap=cmap, transform=ccrs.PlateCarree(), antialiased=False, **kwargs)
    #     if coastlines:
    #         ax.add_feature(cartopy.feature.COASTLINE, edgecolor='white', facecolor='none', linewidth=1.5)
    #     if colorbar:
    #         plt.colorbar(im)
    #     plt.title(title, y=1.05)
    #     plt.show()
    #     return im
    #
    # plot_sphere(sph_feat[..., 0])
    # plot_sphere(sph_feat[..., 1])
    # plot_sphere(sph_feat[..., 2])

