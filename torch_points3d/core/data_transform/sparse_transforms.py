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
        mapping = {"x": 0, "y": 1, "z": 2}
        self._ignored_axis = [mapping[axis] for axis in ignored_axis]
        # Use the rest of axes for flipping.
        self._horz_axes = set(range(self._D)) - set(self._ignored_axis)
        self._p = p

    def __call__(self, data):
        for curr_ax in self._horz_axes:
            if random.random() < self._p:
                coords = data.coords
                coord_max = torch.max(coords[:, curr_ax])
                data.coords[:, curr_ax] = coord_max - coords[:, curr_ax]
        return data

    def __repr__(self):
        return "{}(flip_axis={}, prob={}, is_temporal={})".format(
            self.__class__.__name__, self._horz_axes, self._p, self._is_temporal
        )
