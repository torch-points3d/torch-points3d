
from typing import List
import itertools
import numpy as np
import math
import re
import torch
import scipy
import random
from tqdm.auto import tqdm as tq
from torch.nn import functional as F
from sklearn.neighbors import NearestNeighbors, KDTree
from functools import partial
from torch_geometric.nn import fps, radius, knn, voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos, pool_batch
from torch_scatter import scatter_add, scatter_mean
from torch_cluster import grid_cluster

from torch_points3d.datasets.multiscale_data import MultiScaleData
from torch_points3d.utils.config import is_list
from torch_points3d.utils import is_iterable
from .grid_transform import group_data, GridSampling3D, shuffle_data


class RemoveDuplicateCoords(object):
    """ This transform allow sto remove duplicated coords within ``indices`` from data.
    Takes the average or selects the last point to set the features and labels of each voxel

    Parameters
    ----------
    mode : str
        Option to select how the features and labels for each voxel is computed. Can be ``last`` or ``mean``.
        ``last`` selects the last point falling in a voxel as the representent, ``mean`` takes the average.
    """

    def __init__(self, mode="last"):
        self._mode = mode

    def _process(self, data):
        if self._mode == "last":
            data = shuffle_data(data)

        coords = data.pos
        if "batch" not in data:
            cluster = grid_cluster(coords, torch.tensor([1, 1, 1]))
        else:
            cluster = voxel_grid(coords, data.batch, 1)
        cluster, unique_pos_indices = consecutive_cluster(cluster)

        skip_keys=[]
        if self._mode == "last":
            skip_keys.append("pos")
            data.pos = coords[unique_pos_indices]
        data = group_data(data, cluster, unique_pos_indices, mode=self._mode, skip_keys=skip_keys)
        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in tq(data)]
            data = list(itertools.chain(*data))  # 2d list needs to be flatten
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(mode={})".format(self.__class__.__name__, self._mode)

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
        Option to select how the features and labels for each voxel is computed. Can be ``last`` or ``mean``.
        ``last`` selects the last point falling in a voxel as the representent, ``mean`` takes the average.

    Returns
    -------
    data: Data
        Returns the same data object with only one point per voxel
    """

    def __init__(self, grid_size=None, mode="last"):

        self._grid_size = grid_size
        self._mode = mode

        self._transform = GridSampling3D(grid_size, quantize_coords=True, mode=mode)

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


class RandomCoordsFlip(object):

    def __init__(self, ignored_axis, is_temporal=False, p=0.95):
        """This transform is used to flip sparse coords using a given axis. Usually, it would be x or y

        Parameters
        ----------
        ignored_axis: str
            Axis to be chosen between x, y, z
        is_temporal : bool
            Used to indicate if the pointcloud is actually 4 dimensional

        Returns
        -------
        data: Data
            Returns the same data object with only one point per voxel
        """
        assert 0 <= p <= 1, "p should be within 0 and 1. Higher probability reduce chance of flipping"
        self._is_temporal = is_temporal
        self._D = 4 if is_temporal else 3
        mapping = {'x': 0, 'y': 1, 'z': 2}
        self._ignored_axis = [mapping[axis] for axis in ignored_axis]
        # Use the rest of axes for flipping.
        self._horz_axes = set(range(self._D)) - set(self._ignored_axis)
        self._p = p

    def __call__(self, data):
        for curr_ax in self._horz_axes:
            if random.random() < self._p:
                coords = data.pos
                coord_max = torch.max(coords[:, curr_ax])
                data.pos[:, curr_ax] = coord_max - coords[:, curr_ax]
        return data

    def __repr__(self):
        return "{}(flip_axis={}, prob={}, is_temporal={})"\
            .format(self.__class__.__name__, self._horz_axes, self._p, self._is_temporal)
