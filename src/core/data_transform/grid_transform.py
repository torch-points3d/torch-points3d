import numpy
import re
import logging
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn import voxel_grid


log = logging.getLogger(__name__)


def shuffle_data(data):
    num_points = data.pos.shape[0]
    shuffle_idx = torch.randperm(num_points)
    for key in set(data.keys):
        item = data[key]
        if num_points == item.shape[0]:
            data[key] = item[shuffle_idx]
    return data

def sparse_coords_to_clusters(pos, batch):
    assert pos.dtype == torch.int or pos.dtype == torch.long

    # Build clusters
    if batch is not None:
        pos = torch.cat([batch.unsqueeze(-1), pos.long()], dim=-1)

    unique_pos, cluster = torch.unique(pos, return_inverse=True, dim=0)
    unique_pos_indices = torch.arange(cluster.size(0), dtype=cluster.dtype, device=cluster.device)
    unique_pos_indices = cluster.new_empty(unique_pos.size(0)).scatter_(0, cluster, unique_pos_indices)

    return cluster, unique_pos_indices

def group_data(data, cluster, unique_pos_indices=None, mode="last"):
    """ Group data based on indices in cluster. 
    The option ``mode`` controls how data gets agregated within each cluster.
    
    Parameters
    ----------
    data : Data
        [description]
    cluster : torch.Tensor
        Tensor of the same size as the number of points in data. Each element is the cluster index of that point.
    unique_pos_indices : torch.tensor
        Tensor containing one index per cluster
    mode : str
        Option to select how the features and labels for each voxel is computed. Can be ``last`` or ``mean``.
        ``last`` selects the last point falling in a voxel as the representent, ``mean`` takes the average.
    """

    assert mode in ["mean", "last"]

    num_nodes = data.num_nodes
    for key, item in data:
        if bool(re.search("edge", key)):
            raise ValueError("Edges not supported. Wrong data type.")

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


class GridSampling:
    """ Clusters points into voxels with size :attr:`size`.

    Parameters
    ----------
    grid_size: float
        Size of a voxel (in each dimension).
    to_sparse_coords: bool
        If True, it will convert the points into their associated sparse coordinates within the grid. \
    mode: string:
        The mode can be either `last` or `mean`.
        If mode is `mean`, all the points and their features within a cell will be averaged
        If mode is `last`, one random points per cell will be selected with its associated features
    """

    def __init__(self, size, to_sparse_coords=False, mode="mean"):
        self._grid_size = size
        self._to_sparse_coords = to_sparse_coords
        self._mode = mode

        log.warn("If you need to keep track of the position of your points, use SaveOriginalPosId transform before using GridSampling")

        if self._mode == "last":
            log.warn("The data are going to be shuffled. Be careful that if an attribute doesn't have the size of num_points, it won't be shuffled")

    def _process(self, data):
        if self._mode == "last":
            data = shuffle_data(data)
        
        coords = ((data.pos) / self._grid_size).int()
        batch = data.batch if hasattr(data, "batch") else None
        cluster, unique_pos_indices = sparse_coords_to_clusters(coords, batch)
        
        if self._to_sparse_coords:
            delattr(data, "pos")
        data = group_data(data, cluster, unique_pos_indices, mode=self._mode)
        
        if self._to_sparse_coords:
            data.pos = coords[unique_pos_indices]
        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(grid_size={}, to_sparse_coords={}, mode={}, save_cluster={})".format(self.__class__.__name__, self._grid_size, self._to_sparse_coords, self._mode, self._save_cluster)


class SaveOriginalPosId:
    """ Transform that adds the index of the point to the data object
    This allows us to track this point from the output back to the input data object
    """

    KEY = "origin_id"

    def __call__(self, data):
        setattr(data, self.KEY, torch.arange(0, data.pos.shape[0]))
        return data
