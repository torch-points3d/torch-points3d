
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
from torch_cluster import grid_cluster

from src.datasets.multiscale_data import MultiScaleData
from src.utils.transform_utils import SamplingStrategy
from src.utils.config import is_list
from src.utils import is_iterable
from .grid_transform import group_data, GridSampling, shuffle_data


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