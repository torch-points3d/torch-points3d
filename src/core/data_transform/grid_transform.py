import re
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn import voxel_grid


def group_data(data, cluster, unique_pos_indices, mode="last"):
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
    num_nodes = data.num_nodes
    for key, item in data:
        if bool(re.search("edge", key)):
            raise ValueError("Edges not supported. Wrong data type.")

        if torch.is_tensor(item) and item.size(0) == num_nodes:
            if mode == "last" or key == "batch" or key == SaveOriginalPosId.KEY:
                data[key] = item[unique_pos_indices]
            elif mode == "mean":
                if key == "y":
                    item = F.one_hot(item)
                    item = scatter_add(item, cluster, dim=0)
                    data[key] = item.argmax(dim=-1)
                else:
                    data[key] = scatter_mean(item, cluster, dim=0)
    return data


class GridSampling:
    """ Clusters points into voxels with size :attr:`size`.

    Parameters
    ----------
    size: float
        Size of a voxel (in each dimension).
    start: float
        Start coordinates of the grid (in each dimension). \
        If set to `None`, will be set to the minimum coordinates found in `data.pos`. (default: `None`)
    end: float
        End coordinates of the grid (in each dimension). \
        If set to `None`, will be set to the maximum coordinates found in `data.pos`. (default: `None`)
    num_classes: max number of classes for one hot encoding of y vector
    """

    def __init__(self, size, start=None, end=None, num_classes=-1):
        self.size = size
        self.start = start
        self.end = end
        self.num_classes = num_classes

    def _process(self, data):
        num_nodes = data.num_nodes

        if "batch" not in data:
            batch = data.pos.new_zeros(num_nodes, dtype=torch.long)
        else:
            batch = data.batch

        cluster = voxel_grid(data.pos, batch, self.size, self.start, self.end)
        cluster, perm = consecutive_cluster(cluster)

        return group_data(data, cluster, perm, mode="mean")

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(size={})".format(self.__class__.__name__, self.size)


class SaveOriginalPosId:
    """ Transform that adds the index of the point to the data object
    This allows us to track this point from the output back to the input data object
    """

    KEY = "origin_id"

    def __call__(self, data):
        setattr(data, self.KEY, torch.arange(0, data.pos.shape[0]))
        return data
