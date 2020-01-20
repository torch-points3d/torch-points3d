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
from torch_scatter import scatter_add


import torch
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos, pool_batch
from torch_scatter import scatter_add


class GridSampling(object):
    r"""Samples points depending on :obj:`subsampling_param`.

    Args:
        subsampling_param (int): The subsampling parameters used to map the pointcloud on voxel_grid.
        num_classes (int): If the data contains labels, within key `y`, then it will be used to create label grid pooling. 
    """

    def __init__(self, subsampling_param, num_classes):
        self._subsampling_param = subsampling_param
        self._num_classes = num_classes

    def __call__(self, data):
        num_nodes = data.num_nodes

        pos = data.pos
        batch = data.batch

        pool = voxel_grid(pos, batch, self._subsampling_param)
        pool, _ = consecutive_cluster(pool)

        for key, item in data:
            if bool(re.search('edge', key)):
                continue

            if torch.is_tensor(item) and item.size(0) == num_nodes:
                if key == 'y':
                    one_hot = torch.zeros((item.shape[0], self._num_classes))\
                        .scatter(1, item.unsqueeze(-1), 1)

                    aggr_labels = scatter_add(one_hot, pool, dim=0)
                    data[key] = torch.argmax(aggr_labels, -1)
                else:
                    data[key] = pool_pos(pool, item).to(item.dtype)
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self._subsampling_param)


class Center(object):
    def __call__(self, data):
        data.pos = data.pos - data.pos.mean(axis=0)
        return data

    def __repr__(self):
        return "Center"


class RandomTranslate(object):
    def __init__(self, translate):
        self.translate = translate

    def __call__(self, data):

        t = 2 * (torch.rand(3) - 0.5) * self.translate

        data.pos = data.pos + t
        return data

    def __repr__(self):
        return "Random Translate of translation {}".format(self.translate)


class RandomScale(object):
    def __init__(self, scale_min=1, scale_max=1, is_anisotropic=False):
        if scale_min > scale_max:
            raise ValueError("Scale min must be lesser or equal to Scale max")
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.is_anisotropic = is_anisotropic

    def __call__(self, data):
        scale = self.scale_min + torch.rand(1) * (self.scale_max - self.scale_min)
        if self.is_anisotropic:
            ax = torch.randint(0, 3, 1)
            data.pos[:, ax] = scale * data.pos[:, ax]
        else:
            data.pos = scale * data.pos
        return data

    def __repr__(self):
        return "Random Scale min={}, max={}".format(self.scale_min, self.scale_max)


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
