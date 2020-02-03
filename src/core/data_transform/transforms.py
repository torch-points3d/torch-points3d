import numpy as np
import math
import re
import torch
from torch.nn import functional as F
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from functools import partial
from torch_geometric.nn import fps, radius, knn, voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos, pool_batch
from torch_scatter import scatter_add, scatter_mean


class GridSampling(object):
    r"""Clusters points into voxels with size :attr:`size`.
    Args:
        size (float or [float] or Tensor): Size of a voxel (in each dimension).
        start (float or [float] or Tensor, optional): Start coordinates of the
            grid (in each dimension). If set to :obj:`None`, will be set to the
            minimum coordinates found in :obj:`data.pos`.
            (default: :obj:`None`)
        end (float or [float] or Tensor, optional): End coordinates of the grid
            (in each dimension). If set to :obj:`None`, will be set to the
            maximum coordinates found in :obj:`data.pos`.
            (default: :obj:`None`)
        num_classes (int, optional): number of classes in the dataset (speeds up the computation slightly)
    """

    def __init__(self, size, start=None, end=None, num_classes=-1):
        self.size = size
        self.start = start
        self.end = end
        self.num_classes = num_classes

    def __call__(self, data):
        num_nodes = data.num_nodes

        if "batch" not in data:
            batch = data.pos.new_zeros(num_nodes, dtype=torch.long)
        else:
            batch = data.batch

        cluster = voxel_grid(data.pos, batch, self.size, self.start, self.end)
        cluster, perm = consecutive_cluster(cluster)

        for key, item in data:
            if bool(re.search("edge", key)):
                raise ValueError("GridSampling does not support coarsening of edges")

            if torch.is_tensor(item) and item.size(0) == num_nodes:
                if key == "y":
                    item = F.one_hot(item, num_classes=self.num_classes)
                    item = scatter_add(item, cluster, dim=0)
                    data[key] = item.argmax(dim=-1)
                elif key == "batch":
                    data[key] = item[perm]
                else:
                    data[key] = scatter_mean(item, cluster, dim=0)

        return data

    def __repr__(self):
        return "{}(size={})".format(self.__class__.__name__, self.size)


class RandomSymmetry(object):
    def __init__(self, axis=[False, False, False]):
        self.axis = axis

    def __call__(self, data):

        for i, ax in enumerate(self.axis):
            if ax:
                if torch.rand(1) < 0.5:
                    data.pos[:, i] *= -1
        return data

    def __repr__(self):
        return "Random symmetry of axes: x={}, y={}, z={}".format(*self.axis)


class RandomNoise(object):
    def __init__(self, sigma=0.01, clip=0.05):
        """
        simple isotropic additive gaussian noise
        """
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data):
        noise = self.sigma * torch.randn(data.pos.shape)
        noise = noise.clamp(-self.clip, self.clip)
        data.pos = data.pos + noise
        return data

    def __repr__(self):
        return "Random noise of sigma={}".format(self.sigma)


def euler_angles_to_rotation_matrix(theta):
    """
    """
    R_x = torch.tensor(
        [[1, 0, 0], [0, torch.cos(theta[0]), -torch.sin(theta[0])], [0, torch.sin(theta[0]), torch.cos(theta[0])]]
    )

    R_y = torch.tensor(
        [[torch.cos(theta[1]), 0, torch.sin(theta[1])], [0, 1, 0], [-torch.sin(theta[1]), 0, torch.cos(theta[1])]]
    )

    R_z = torch.tensor(
        [[torch.cos(theta[2]), -torch.sin(theta[2]), 0], [torch.sin(theta[2]), torch.cos(theta[2]), 0], [0, 0, 1]]
    )

    R = torch.mm(R_z, torch.mm(R_y, R_x))
    return R


class RandomRotation(object):
    def __init__(self, mode="vertical"):
        """
        random rotation: either
        """
        self.mode = mode

    def __call__(self, data):

        theta = torch.zeros(3)
        if self.mode == "vertical":
            theta[2] = torch.rand(1) * 2 * torch.tensor(math.pi)
        elif self.mode == "all":
            theta = torch.rand(3) * 2 * torch.tensor(math.pi)
        else:
            raise NotImplementedError("this kind of rotation ({}) " "is not yet available".format(self.mode))
        R = euler_angles_to_rotation_matrix(theta)
        data.pos = torch.mm(data.pos, R.t())
        return data

    def __repr__(self):
        return "Random rotation of mode {}".format(self.mode)


class MeshToNormal(object):
    """ Computes mesh normals (IN PROGRESS)
    """

    def __init__(self):
        pass

    def __call__(self, data):
        if hasattr(data, "face"):
            pos = data.pos
            face = data.face
            vertices = [pos[f] for f in face]
            normals = torch.cross(vertices[0] - vertices[1], vertices[0] - vertices[2], dim=1)
            normals = F.normalize(normals)
            data.normals = normals
        return data

    def __repr__(self):
        return "{}".format(self.__class__.__name__)


class MultiScaleTransform(object):
    """ Pre-computes a sequence of downsampling / neighboorhood search
    """

    def __init__(self, strategies, precompute_multi_scale=False):
        self.strategies = strategies
        self.precompute_multi_scale = precompute_multi_scale
        if self.precompute_multi_scale and not bool(strategies):
            raise Exception("Strategies are empty and precompute_multi_scale is set to True")
        self.num_layers = len(self.strategies.keys())

    @staticmethod
    def __inc__wrapper(func, special_params):
        def new__inc__(key, num_nodes, special_params=None, func=None):
            if key in special_params:
                return special_params[key]
            else:
                return func(key, num_nodes)

        return partial(new__inc__, special_params=special_params, func=func)

    def __call__(self, data: Data):
        if self.precompute_multi_scale:
            # Compute sequentially multi_scale indexes on cpu
            special_params = {}
            pos = data.pos
            batch = torch.zeros((pos.shape[0],), dtype=torch.long)
            for index in range(self.num_layers):
                sampler, neighbour_finder = self.strategies[index]
                idx = sampler(pos, batch)
                row, col = neighbour_finder(pos, pos[idx], batch, batch[idx])
                edge_index = torch.stack([col, row], dim=0)

                index_name = "index_{}".format(index)
                edge_name = "edge_index_{}".format(index)

                setattr(data, index_name, idx)
                setattr(data, edge_name, edge_index)

                num_nodes_for_edge_index = torch.from_numpy(np.array([pos.shape[0], pos[idx].shape[0]])).unsqueeze(-1)

                special_params[index_name] = num_nodes_for_edge_index[0]

                special_params[edge_name] = num_nodes_for_edge_index
                pos = pos[idx]
                batch = batch[idx]
            func_ = self.__inc__wrapper(data.__inc__, special_params)
            setattr(data, "__inc__", func_)
        return data

    def __repr__(self):
        return "{}".format(self.__class__.__name__)
