
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay

from .ply_c import libply_c

class GeometricFeatures(object):
    r"""Compute geometric features (linearity, planarity, scattering and verticality)
    """
    def __init__(self, k_nn_local: int = 50, k_nn_adj: int = 10):
        self.k_nn_local = k_nn_local
        self.k_nn_adj = k_nn_adj

    def __call__(self, data):
        pos = data.pos
        nn = NearestNeighbors(n_neighbors=self.k_nn_local+1, algorithm='kd_tree').fit(pos)
        _, neighboors = nn.kneighbors(pos)
        geof = libply_c.compute_geof(pos, neighboors, self.k_nn_local).astype('float32')
        geof[:, 3] = 2. * geof[:, 3]
        data.geof = geof
        return data

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.k_nn_local, self.k_nn_adj)

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
            #Compute sequentially multi_scale indexes on cpu
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