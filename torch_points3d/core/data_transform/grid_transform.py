from typing import *
import numpy as np
import numpy
import scipy
import re
import torch
import logging
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn import voxel_grid
from torch_geometric.data import Data
from torch_cluster import grid_cluster

log = logging.getLogger(__name__)


def shuffle_data(data):
    num_points = data.pos.shape[0]
    shuffle_idx = torch.randperm(num_points)
    for key in set(data.keys):
        item = data[key]
        if torch.is_tensor(item) and num_points == item.shape[0]:
            data[key] = item[shuffle_idx]
    return data


def group_data(data, cluster=None, unique_pos_indices=None, mode="last", skip_keys=[]):
    """ Group data based on indices in cluster.
    The option ``mode`` controls how data gets agregated within each cluster.

    Parameters
    ----------
    data : Data
        [description]
    cluster : torch.Tensor
        Tensor of the same size as the number of points in data. Each element is the cluster index of that point.
    unique_pos_indices : torch.tensor
        Tensor containing one index per cluster, this index will be used to select features and labels
    mode : str
        Option to select how the features and labels for each voxel is computed. Can be ``last`` or ``mean``.
        ``last`` selects the last point falling in a voxel as the representent, ``mean`` takes the average.
    skip_keys: list
        Keys of attributes to skip in the grouping
    """

    assert mode in ["mean", "last"]
    if mode == "mean" and cluster is None:
        raise ValueError("In mean mode the cluster argument needs to be specified")
    if mode == "last" and unique_pos_indices is None:
        raise ValueError("In last mode the unique_pos_indices argument needs to be specified")

    num_nodes = data.num_nodes
    for key, item in data:
        if bool(re.search("edge", key)):
            raise ValueError("Edges not supported. Wrong data type.")
        if key in skip_keys:
            continue

        if torch.is_tensor(item) and item.size(0) == num_nodes:
            if mode == "last" or key == "batch" or key == SaveOriginalPosId.KEY:
                data[key] = item[unique_pos_indices]
            elif mode == "mean":
                if key == "y":
                    item_min = item.min()
                    item = F.one_hot(item - item_min)
                    item = scatter_add(item, cluster, dim=0)
                    data[key] = item.argmax(dim=-1) + item_min
                else:
                    data[key] = scatter_mean(item, cluster, dim=0)
    return data


class GridSampling3D:
    """ Clusters points into voxels with size :attr:`size`.
    Parameters
    ----------
    size: float
        Size of a voxel (in each dimension).
    quantize_coords: bool
            If True, it will convert the points into their associated sparse coordinates within the grid. \
    mode: string:
        The mode can be either `last` or `mean`.
        If mode is `mean`, all the points and their features within a cell will be averaged
        If mode is `last`, one random points per cell will be selected with its associated features
    """

    def __init__(self, size, quantize_coords=False, mode="mean", verbose=False):
        self._grid_size = size
        self._quantize_coords = quantize_coords
        self._mode = mode
        if verbose:
            log.warning(
                "If you need to keep track of the position of your points, use SaveOriginalPosId transform before using GridSampling3D"
            )

            if self._mode == "last":
                log.warning(
                    "The tensors within data will be shuffled each time this transform is applied. Be careful that if an attribute doesn't have the size of num_points, it won't be shuffled"
                )

    def _process(self, data):
        if self._mode == "last":
            data = shuffle_data(data)

        coords = ((data.pos) / self._grid_size).int()
        if "batch" not in data:
            cluster = grid_cluster(coords, torch.tensor([1, 1, 1]))
        else:
            cluster = voxel_grid(coords, data.batch, 1)
        cluster, unique_pos_indices = consecutive_cluster(cluster)

        skip_keys = []
        if self._quantize_coords:
            skip_keys.append("pos")

        data = group_data(data, cluster, unique_pos_indices, mode=self._mode, skip_keys=skip_keys)

        if self._quantize_coords:
            data.pos = coords[unique_pos_indices]

        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(grid_size={}, quantize_coords={}, mode={})".format(
            self.__class__.__name__, self._grid_size, self._quantize_coords, self._mode
        )


class SaveOriginalPosId:
    """ Transform that adds the index of the point to the data object
    This allows us to track this point from the output back to the input data object
    """

    KEY = "origin_id"

    def _process(self, data):
        if hasattr(data, self.KEY):
            return data

        setattr(data, self.KEY, torch.arange(0, data.pos.shape[0]))
        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data


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

    def __init__(self, apply_distorsion: bool = True, granularity: List = [0.2, 0.4]):
        self._apply_distorsion = apply_distorsion
        self._granularity = list(granularity)

    @staticmethod
    def elastic_distortion(coords, granularity):
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)[0]

        # Create Gaussian noise tensor of the size given by granularity.
        dim = coords.shape[-1]
        denom = torch.Tensor([np.random.uniform(granularity[0], granularity[1]) for _ in range(dim)])
        noise_dim = ((coords - coords_min).max(0)[0] // denom).int() + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        granularity_shift = granularity[1]
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity_shift, coords_min + granularity_shift * (noise_dim - 2), noise_dim
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        return (coords + torch.Tensor(interp(coords))).int()

    def __call__(self, data):
        if self._apply_distorsion:
            if np.random.uniform(0, 1) < 0.5:
                data.pos = ElasticDistortion.elastic_distortion(data.pos, torch.Tensor(self._granularity))
        return data

    def __repr__(self):
        return "{}(apply_distorsion={}, granularity={})".format(
            self.__class__.__name__, self._apply_distorsion, self._granularity
        )
