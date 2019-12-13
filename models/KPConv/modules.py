import inspect
import sys

from enum import Enum
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import math
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LeakyReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, MessagePassing
from torch.nn.parameter import Parameter
from .kernel_utils import kernel_point_optimization_debug
from models.core_modules import *
from models.base_model import BaseFactory

class KPConvModels(Enum):
    KPCONV = 0
    RESIDUALBKPCONV = 1

class KPConvFactory(BaseFactory):
    def get_module_from_index(self, index, flow=None):
        if flow is None:
            raise NotImplementedError
        
        if flow.upper() == "UP":
            return getattr(self.modules_lib, self.module_name_up, None)
        
        if flow.upper() == "DOWN":
            if self.module_name_down.upper() == str(KPConvModels.KPCONV.name):
                return KPConv
            elif self.module_name_down.upper() == str(KPConvModels.RESIDUALBKPCONV.name):
                if index == 0:
                    return KPConv
                else:
                    return ResidualBKPConv
        
        raise NotImplementedError

class PointKernel(MessagePassing):

    '''
    Implements KPConv: Flexible and Deformable Convolution for Point Clouds from 
    https://arxiv.org/abs/1904.08889
    '''

    def __init__(self, num_points, in_features, out_features, radius=1, kernel_dim=3, fixed='center', ratio=1, KP_influence='linear'):
        super(PointKernel, self).__init__()
        # PointKernel parameters
        self.in_features = in_features
        self.out_features = out_features
        self.num_points = num_points
        self.radius = radius
        self.kernel_dim = kernel_dim
        self.fixed = fixed
        self.ratio = ratio
        self.KP_influence = KP_influence

        # Radius of the initial positions of the kernel points
        self.KP_extent = radius / 1.5

        # Point position in kernel_dim
        self.kernel = Parameter(torch.Tensor(1, num_points, kernel_dim))

        # Associated weights
        self.kernel_weight = Parameter(torch.Tensor(num_points, in_features, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.kernel_weight, a=math.sqrt(5))

        # Init the kernel using attrative + repulsion forces
        kernel, _ = kernel_point_optimization_debug(self.radius, self.num_points, num_kernels=1,
                                                    dimension=self.kernel_dim, fixed=self.fixed, ratio=self.ratio, verbose=False)
        self.kernel.data = torch.from_numpy(kernel)

    def forward(self, x, pos, edge_index):
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_j, pos_i, pos_j):

        if x_j is None:
            x_j = pos_j

        # Center every neighborhood [SUM n_neighbors(n_points), dim]
        neighbors = (pos_j - pos_i)

        # Number of points
        n_points = neighbors.shape[0]

        # Get points kernels
        K_points = self.kernel

        # Get all difference matrices [SUM n_neighbors(n_points), n_kpoints, dim]
        neighbors = neighbors.unsqueeze(1)

        differences = neighbors - K_points.float().view((-1, 3)).unsqueeze(0)
        sq_distances = (differences**2).sum(-1)

        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)

        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = 1. - (sq_distances / (self.KP_extent ** 2))
            all_weights[all_weights < 0] = 0.0
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        neighbors_1nn = torch.argmin(sq_distances, dim=-1)
        weights = all_weights.gather(1, neighbors_1nn.unsqueeze(-1))

        K_weights = self.kernel_weight
        K_weights = torch.index_select(K_weights, 0, neighbors_1nn.view(-1)
                                       ).view((n_points, self.in_features, self.out_features))

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        features = x_j

        # Apply distance weights [n_points, n_kpoints, in_fdim]
        weighted_features = torch.einsum("nb, nc -> nc", weights, features)

        # Apply network weights [n_kpoints, n_points, out_fdim]
        out = torch.einsum("na, nac -> nc", weighted_features, K_weights)
        out = out.view(-1, self.out_features)
        #import pdb; pdb.set_trace()
        return out

    def update(self, aggr_out, pos):
        return aggr_out

    def __repr__(self):
                # PointKernel parameters
        return "PointKernel({}, {}, {}, {}, {})".format(self.in_features, self.out_features, self.num_points, self.radius, self.KP_influence)

class KPConv(BaseConvolution):
    def __init__(self, ratio=None, radius=None, down_conv_nn=None, num_points=16, *args, **kwargs):
        super(KPConv, self).__init__(ratio, radius)

        in_features, out_features = down_conv_nn

        # KPCONV arguments
        self.in_features = in_features
        self.out_features = out_features
        self.num_points = num_points

        self._conv = PointKernel(self.num_points, self.in_features, self.out_features, radius=self.radius)

    @property
    def conv(self):
        return self._conv

class ResidualBKPConv(nn.Module):
    def __init__(self, ratio=None, radius=None, down_conv_nn=None, num_points=16, *args, **kwargs):
        super(ResidualBKPConv, self).__init__()
        
        self.ratio = ratio
        self.radius = radius
        self.max_num_neighbors = kwargs.get("max_num_neighbors", 64)
        
        in_features, out_features = down_conv_nn

        # KPCONV arguments
        self.in_features = in_features
        self.out_features = out_features
        self.num_points = num_points

        self.pre_mlp = nn.Linear(self.in_features, self.in_features // 2)
        self._conv = PointKernel(self.num_points, self.in_features // 2, self.in_features // 2, radius=self.radius)
        self.post_mlp = nn.Linear(self.in_features // 2, out_features)

        self.shortcut_mlp = nn.Linear(self.in_features, self.out_features)

    @property
    def conv(self):
        return self._conv

    def forward(self, data):
        x, pos, batch = data
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.radius, batch, batch[idx],
                          max_num_neighbors=self.max_num_neighbors)
        edge_index = torch.stack([col, row], dim=0)
        
        x_side = self.pre_mlp(x)
        x_side = self.conv(x_side, (pos, pos[idx]), edge_index)
        x_side = self.post_mlp(x)

        x_shortcut = self.shortcut_mlp(x[idx])

        pos, batch = pos[idx], batch[idx]
        data = (x_side + x_shortcut, pos, batch)
        return data

class SimpleUpsampleKPConv(BaseConvolution):
    def __init__(self, ratio=None, radius=None, up_conv_nn=None, mlp_nn=None, num_points=16, *args, **kwargs):
        super(SimpleUpsampleKPConv, self).__init__(ratio, radius)

        in_features, out_features = up_conv_nn

        # KPCONV arguments
        self.in_features = in_features
        self.out_features = out_features
        self.num_points = num_points

        self._conv = PointKernel(self.num_points, self.in_features, self.out_features, radius=self.radius)

        self.nn = MLP(mlp_nn, activation=LeakyReLU(0.2))

    @property
    def conv(self):
        return self._conv

    def forward(self, data):
        x, pos, batch, x_skip, pos_skip, batch_skip = data
        row, col = radius(pos, pos_skip, self.radius, batch, batch_skip,
                          max_num_neighbors=self.max_num_neighbors)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos_skip), edge_index)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        data = (x, pos_skip, batch_skip)
        return data
