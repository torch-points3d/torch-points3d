
from typing import List
import itertools
import numpy as np
import math
import re
import torch
import random
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
from torch_geometric.data import Data, Batch
from tqdm import tqdm as tq

from src.utils import is_iterable
from src.modules.MinkowskiEngine import to_sparse_input, quantize_data, shuffle_data


class ShuffleData(object):
    """This transform allow to shuffle tensors within data
    """

    def _process(self, data):
        return shuffle_data(data)

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in tq(data)]
            data = list(itertools.chain(*data))  # 2d list needs to be flatten
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)

class RemoveDuplicateCoords(object):
    """This transform allow to remove duplicated coords within `indices` from data

    Parameters
    ----------
    shuffle: bool
        Wether True, the data will be suffled before being removed
    """

    def __init__(self, shuffle=False):
        self._shuffle = shuffle
        if self._shuffle:
            self._shuffle_transform = ShuffleData()

    def _process(self, data):
        if self._shuffle:
            data = self._shuffle_transform(data)
        return quantize_data(data,mode="last")

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in tq(data)]
            data = list(itertools.chain(*data))  # 2d list needs to be flatten
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(shuffle={})".format(self.__class__.__name__, self._shuffle)

class ToSparseInput(object):
    """This transform allows to prepare data for sparse model as SparseConv / Minkowski Engine

    Parameters
    ----------
    grid_size: float
        Grid voxel size
    save_delta: bool
        If True, the displacement tensor from closest grid to a given point would be saved
    save_delta_norm: bool
        If True, the norm tensor from closest grid to a given point would be saved
    apply_mean: bool
        If True, apply the mean over the points within a cell and associate the value to the grid point.
    """

    def __init__(self, grid_size=None, save_delta: bool=False, save_delta_norm:bool=False, mode="last"):
        self._grid_size = grid_size
        self._save_delta = save_delta
        self._save_delta_norm = save_delta_norm
        self._mode = mode

    def _process(self, data):
        return to_sparse_input(data, self._grid_size, save_delta=self._save_delta, mode=self._mode)

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in tq(data)]
            data = list(itertools.chain(*data))  # 2d list needs to be flatten
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(grid_size={}, save_delta={}, save_delta_norm={}, remove_duplicates={}, apply_mean={})".format(self.__class__.__name__, self._grid_size, self._save_delta, self._save_delta_norm, self._remove_duplicates, self._apply_mean)