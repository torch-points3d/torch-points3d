
from typing import List
import itertools
import numpy as np
import math
import re
import torch
import random
from tqdm import tqdm as tq
from torch.nn import functional as F
from sklearn.neighbors import NearestNeighbors, KDTree
from functools import partial
from torch_geometric.nn import fps, radius, knn, voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos, pool_batch
from torch_scatter import scatter_add, scatter_mean

from src.datasets.multiscale_data import MultiScaleData
from src.utils.transform_utils import SamplingStrategy
from src.utils.config import is_list
from src.utils import is_iterable
from .grid_transform import group_data, GridSampling, shuffle_data, sparse_coords_to_clusters



class ToSparseInput(object):
    """This transform allows to prepare data for sparse model as SparseConv / Minkowski Engine.
    It does the following things:

    - Puts ``pos`` on a fixed integer grid based on grid size
    - Keeps one point per grid cell. The strategy for defining the feature nad label at that point depends on the ``mode`` option

    Parameters
    ----------
    grid_size: float
        Grid voxel size
    mode : str
        Option to select how the features and labels for each voxel are computed. Can be ``last`` or ``mean``.
        ``last`` selects the last point falling in a voxel as the representent,
        ``mean`` takes the average.

    Returns
    -------
    data: Data
        Returns the same data object with only one point per voxel
    """

    def __init__(self, grid_size=None, mode="last"):

        self._grid_size = grid_size
        self._mode = mode

        self._transform = GridSampling(grid_size, quantize_coords=True, mode=mode)

    def _process(self, data):
        return self._transform(data)


    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in tq(data)]
            data = list(itertools.chain(*data))  # 2d list needs to be flatten
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(grid_size={}, mode={})"\
            .format(self.__class__.__name__, self._grid_size, self._mode)


class ElasticDistortion:

    """Apply elastic distortion on sparse coordinate space.

    Parameters
    ----------
    granularity: float
        Size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: bool
        Noise multiplier

    Returns
    -------
    data: Data
        Returns the same data object with distorted grid
    """

    def __init__(self, apply_distorsion:bool, granularity: list = [0.2, 0.4], magnitude: list = [0.8, 1.6]):
        self._apply_distorsion = apply_distorsion
        self._granularity = granularity
        self._magnitude = magnitude

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
        blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
        blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                    (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, coords, feats, labels):
        if self._apply_distorsion:
            if np.random.uniform(0, 1) < .5:
                data.pos = ElasticDistortion.elastic_distortion(data.pos, granularity, magnitude)
        return data

    def __repr__(self):
        return "{}(apply_distorsion={}, granularity={}, magnitude={})"\
        .format(self.__class__.__name__, self._apply_distorsion, self._granularity, self._magnitude)
