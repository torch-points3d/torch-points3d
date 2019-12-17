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
from torch_geometric.utils import scatter_
from models.core_sampling_and_search import RadiusNeighbourFinder, FPSSampler
from .kernels import PointKernel, LightDeformablePointKernel

class KPConvModels(Enum):
    KPCONV = 0
    RESIDUALBKPCONV = 1
    RESIDUALUPSAMPLEBKPCONV = 2
    LIGHTDEFORMABLEKPCONV = 3

class KPConvFactory(BaseFactory):
    def get_module_from_index(self, index, flow=None):
        if flow is None:
            raise NotImplementedError
        
        if flow.upper() == "UP":
            return getattr(self.modules_lib, self.module_name_up, None)
        
        if flow.upper() == "DOWN":
            if self.module_name_down.upper() == str(KPConvModels.KPCONV.name):
                return KPConv
            elif self.module_name_down.upper() == str(KPConvModels.LIGHTDEFORMABLEKPCONV.name):
                return LightDeformableKPConv
            elif self.module_name_down.upper() == str(KPConvModels.RESIDUALBKPCONV.name):
                if index == 0:
                    return KPConv
                else:
                    return ResidualBKPConv
        raise NotImplementedError

class LightDeformableKPConv(BaseConvolution):
    def __init__(self, ratio=None, radius=None, down_conv_nn=None, num_points=16, *args, **kwargs):
        super(LightDeformableKPConv, self).__init__(FPSSampler(ratio), RadiusNeighbourFinder(radius), *args, **kwargs)

        self.ratio = ratio
        self.radius = radius
        
        in_features, out_features = down_conv_nn

        # KPCONV arguments
        self.in_features = in_features
        self.out_features = out_features
        self.num_points = num_points

        self._conv = LightDeformablePointKernel(self.num_points, self.in_features, self.out_features, radius=self.radius)

    @property
    def conv(self):
        return self._conv

class KPConv(BaseConvolution):
    def __init__(self, ratio=None, radius=None, down_conv_nn=None, num_points=16, *args, **kwargs):
        super(KPConv, self).__init__(FPSSampler(ratio), RadiusNeighbourFinder(radius), *args, **kwargs)

        self.ratio = ratio
        self.radius = radius
        
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
        x_side = self.post_mlp(x_side)

        x_shortcut = self.shortcut_mlp(x[idx])

        pos, batch = pos[idx], batch[idx]
        data = (x_side + x_shortcut, pos, batch)
        return data

class SimpleUpsampleKPConv(BaseConvolution):
    def __init__(self, ratio=None, radius=None, up_conv_nn=None, mlp_nn=None, num_points=16, *args, **kwargs):
        super(SimpleUpsampleKPConv, self).__init__(FPSSampler(ratio), RadiusNeighbourFinder(radius), *args, **kwargs)

        in_features, out_features = up_conv_nn

        # KPCONV arguments
        self.radius = radius
        self.in_features = in_features
        self.out_features = out_features
        self.num_points = num_points

        self._conv = PointKernel(self.num_points, self.in_features, self.out_features, radius=self.radius)

        self.nn = MLP(mlp_nn, activation=LeakyReLU(0.2))

    @property
    def conv(self):
        return self._conv

class ResidualUpsampleBKPConv(nn.Module):
    def __init__(self, ratio=None, radius=None, up_conv_nn=None, mlp_nn=None, num_points=16, *args, **kwargs):
        super(ResidualUpsampleBKPConv, self).__init__()

        self.ratio = ratio
        self.radius = radius
        self.max_num_neighbors = kwargs.get("max_num_neighbors", 64)
        
        in_features, out_features = up_conv_nn

        # KPCONV arguments
        self.in_features = in_features
        self.out_features = out_features
        self.num_points = num_points

        self.pre_mlp = nn.Linear(self.in_features, self.in_features // 4)
        self._conv = PointKernel(self.num_points, self.in_features  // 4, self.in_features // 4, radius=self.radius)
        self.post_mlp = nn.Linear(self.in_features // 4, out_features)

        self.shortcut_mlp = nn.Linear(self.in_features, self.out_features)

        self.nn = MLP(mlp_nn, activation=LeakyReLU(0.2))

    @property
    def conv(self):
        return self._conv

    def forward(self, data):
        x, pos, batch, x_skip, pos_skip, batch_skip = data
        row, col = radius(pos, pos_skip, self.radius, batch, batch_skip,
                          max_num_neighbors=self.max_num_neighbors)
        edge_index = torch.stack([col, row], dim=0)

        x_side = self.pre_mlp(x)
        x_side = self.conv(x_side, (pos, pos_skip), edge_index)
        x_side = self.post_mlp(x_side)

        x_shortcut = self.shortcut_mlp(x)
        x_shortcut = torch.index_select(x_shortcut, 0, col)
        x_shortcut = scatter_("add", x_shortcut, row)

        x = x_side + x_shortcut

        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        data = (x, pos_skip, batch_skip)
        return data