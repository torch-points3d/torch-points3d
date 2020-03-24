
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
from .grid_transform import group_data


def shuffle_data(data):
    num_points = data.pos.shape[0]
    shuffle_idx = torch.randperm(num_points)
    for key in set(data.keys):
        item = data[key]
        if num_points == item.shape[0]:
            data[key] = item[shuffle_idx]
    return data


def quantize_data(data, mode="last"):
    """ Creates the quantized version of a data object in which ``pos`` has
    already been quantized. It either averages all points in the same cell or picks
    the last one as being the representent for that cell.

    Parameters
    ----------
    data : Data
        data object in which ``pos`` has already been quantized. ``pos`` must be
        a ``torch.Tensor[int]`` or ``torch.Tensor[long]``
    mode : str
        Option to select how the features and labels for each voxel is computed. Can be ``last`` or ``mean``.
        ``last`` selects the last point falling in a voxel as the representent, ``mean`` takes the average.

    Returns
    -------
    data: Data
        Returns the same data object with only one point per voxel
    """
    assert data.pos.dtype == torch.int or data.pos.dtype == torch.long

    # Build clusters
    if hasattr(data, "batch"):
        pos = torch.cat([data.batch.unsqueeze(-1), data.pos], dim=-1)
    else:
        pos = data.pos.clone()
    unique_pos, cluster = torch.unique(pos, return_inverse=True, dim=0)
    unique_pos_indices = torch.arange(cluster.size(0), dtype=cluster.dtype, device=cluster.device)
    unique_pos_indices = cluster.new_empty(unique_pos.size(0)).scatter_(0, cluster, unique_pos_indices)

    # Agregate features within the same voxel
    data = group_data(data, cluster, unique_pos_indices, mode=mode)
    return data


def to_sparse_input(
    data, grid_size, save_delta=True, save_delta_norm=True, mode="last", quantizing_func=torch.floor,
):
    if quantizing_func not in [torch.floor, torch.ceil, torch.round]:
        raise Exception("quantizing_func should be floor, ceil, round")

    if grid_size is None or grid_size <= 0:
        raise Exception("Grid size should be provided and greater than 0")

    # Quantize positions
    raw_pos = data.pos.clone()
    data.pos = quantizing_func(data.pos / grid_size).int()

    # Add delta as a feature
    if save_delta or save_delta_norm:
        normalised_delta = compute_sparse_delta(raw_pos, data.pos, grid_size, quantizing_func)
        if save_delta:
            data.delta = normalised_delta
        if save_delta_norm:
            data.delta_norm = torch.norm(normalised_delta, p=2, dim=-1)  # normalise between -1 and 1 (roughly)

    # Agregate
    data = quantize_data(data, mode=mode)
    return data


def compute_sparse_delta(raw_pos, quantized_pos, grid_size, quantizing_func):
    """ Computes the error between the raw position and the quantized position
    Error is normalised between -1 and 1

    Parameters
    ----------
    raw_pos : torch.Tensor
    quantized_pos : torch.Tensor
    quantizing_func : func


    Returns
    -------
    torch.Tensor
        Error normalized between -1 and 1
    """
    delta = raw_pos - quantized_pos
    shift = 0
    if quantizing_func == torch.ceil:
        shift = 1
    elif quantizing_func == torch.floor:
        shift = -1

    return 2 * delta / grid_size + shift  # normalise between -1 and 1 (roughly)


class RemoveDuplicateCoords(object):
    """ This transform allow sto remove duplicated coords within ``indices`` from data.
    Selects the last point within each voxel to set the features and labels.

    Parameters
    ----------
    shuffle: bool
        If True, the data will be suffled before removing the extra points
    """

    def __init__(self, shuffle=False):
        self._shuffle = shuffle

    def _process(self, data):
        if self._shuffle:
            data = shuffle_data(data)
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
    """This transform allows to prepare data for sparse model as SparseConv / Minkowski Engine.
    It does the following things:

    - Puts ``pos`` on a fixed integer grid based on grid size
    - Keeps one point per grid cell. The strategy for defining the feature nad label at that point depends on the ``mode`` option

    Parameters
    ----------
    grid_size: float
        Grid voxel size
    save_delta: bool
        If True, the displacement tensor from closest grid to a given point would be saved. It is normalised between -1 and 1.
        New feature: ``delta``
    save_delta_norm: bool
        If True, the norm tensor from closest grid to a given point would be saved. It is normalised between -1 and 1.
        New feature: ``delta_norm``
    mode : str
        Option to select how the features and labels for each voxel are computed. Can be ``last`` or ``mean``.
        ``last`` selects the last point falling in a voxel as the representent, ``mean`` takes the average.

    Returns
    -------
    data: Data
        Returns the same data object with only one point per voxel
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
        return "{}(grid_size={}, save_delta={}, save_delta_norm={}, mode={})"\
            .format(self.__class__.__name__, self._grid_size, self._save_delta, self._save_delta_norm, self._mode)