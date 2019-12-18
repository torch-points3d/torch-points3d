
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay


class MeshToNormal(object):
    r"""Compute geometric features (linearity, planarity, scattering and verticality)
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
        return '{}'.format(self.__class__.__name__)


class MultiScaleTransform(object):
    r"""Compute geometric features (linearity, planarity, scattering and verticality)
    """

    def __init__(self, strategies, precompute_multi_scale=False):
        self.strategies = strategies
        self.precompute_multi_scale = precompute_multi_scale
        if self.precompute_multi_scale and not bool(strategies):
            raise Exception('Strategies are empty and precompute_multi_scale is set to True')
        self.num_layers = len(self.strategies.keys())

    def __call__(self, data):
        if self.precompute_multi_scale:
            # Compute sequentially multi_scale indexes on cpu
            pos = data.pos
            batch = data.batch
            for index in range(self.num_layers):
                sampler, neighbour_finder = self.strategies[index]
                idx = sampler(pos, batch)
                row, col = neighbour_finder(pos, pos[idx], batch, batch[idx])
                edge_index = torch.stack([col, row], dim=0)
                setattr(data, "idx_{}".format(index), idx)
                setattr(data, "edge_index_{}".format(index), edge_index)
                pos = pos[idx]
                batch = batch[idx]
        return data

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
