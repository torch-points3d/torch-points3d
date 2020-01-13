import inspect
import sys

from enum import Enum
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import math
from torch.nn import (
    Sequential as Seq,
    Linear as Lin,
    ReLU,
    LeakyReLU,
    BatchNorm1d as BN,
)
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, MessagePassing
from torch.nn.parameter import Parameter
from torch_geometric.utils import scatter_
from models.core_sampling_and_search import RadiusNeighbourFinder, FPSSampler
from .kernels import PointKernel, LightDeformablePointKernel, PointKernelPartialDense
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_max


# Kernel Point Convolution in Pytorch
# Adaption from https://github.com/humanpose1/KPConvTorch/blob/master/models/layers.py
from .kernel_utils import load_kernels as create_kernel_points
from .convolution_ops import KPConv_deform_ops
from .kernels import PointKernel, LightDeformablePointKernel
from .kernel_utils import kernel_point_optimization_debug
from models.core_sampling_and_search import RadiusNeighbourFinder, FPSSampler
from models.core_modules import *
from models.unet_base import BaseFactory
from .partial_dense_modules import *


class KPConvModels(Enum):
    KPCONV = 0
    RESIDUALBKPCONV = 1
    RESIDUALUPSAMPLEBKPCONV = 2
    LIGHTDEFORMABLEKPCONV = 3
    KPCONVPARTIALDENSE = 4
    RESNETBOTTLENECKPARTIALDENSE = 5


class KPConvFactory(BaseFactory):
    def get_module(self, index, flow=None):
        if flow is None:
            raise NotImplementedError

        if flow.upper() == "UP":
            return getattr(self.modules_lib, self.module_name_up, None)

        if flow.upper() == "DOWN":
            if self.module_name_down.upper() == str(KPConvModels.RESIDUALBKPCONV.name):
                if index == 0:
                    return KPConv
                else:
                    return ResidualBKPConv
            else:
                return getattr(self.modules_lib, self.module_name_down, None)

        raise NotImplementedError


####################### BUILT WITH BaseConvolutionDown ############################


class LightDeformableKPConv(BaseConvolutionDown):
    def __init__(self, ratio=None, radius=None, down_conv_nn=None, kp_points=16, nb_feature=0, *args, **kwargs):
        super(LightDeformableKPConv, self).__init__(FPSSampler(ratio), RadiusNeighbourFinder(radius), *args, **kwargs)

        self.ratio = ratio
        self.radius = radius

        in_features, out_features = down_conv_nn

        # KPCONV arguments
        self.in_features = in_features
        self.out_features = out_features
        self.kp_points = kp_points

        self._conv = LightDeformablePointKernel(self.kp_points, self.in_features, self.out_features, radius=self.radius)

    def conv(self, x, pos, edge_index, batch):
        return self._conv(x, pos, edge_index)


class KPConv(BaseConvolutionDown):
    def __init__(self, ratio=None, radius=None, down_conv_nn=None, kp_points=16, nb_feature=0, *args, **kwargs):
        super(KPConv, self).__init__(FPSSampler(ratio), RadiusNeighbourFinder(radius), *args, **kwargs)

        self.ratio = ratio
        self.radius = radius

        in_features, out_features = down_conv_nn

        # KPCONV arguments
        self.in_features = in_features
        self.out_features = out_features
        self.kp_points = kp_points

        self._conv = PointKernel(self.kp_points, self.in_features, self.out_features, radius=self.radius)

    def conv(self, x, pos, edge_index, batch):
        return self._conv(x, pos, edge_index)


class ResidualBKPConv(BaseConvolutionDown):
    def __init__(self, ratio=None, radius=None, down_conv_nn=None, kp_points=16, nb_feature=0, *args, **kwargs):
        super(ResidualBKPConv, self).__init__(FPSSampler(ratio), RadiusNeighbourFinder(radius), *args, **kwargs)

        self.ratio = ratio
        self.radius = radius
        self.max_num_neighbors = kwargs.get("max_num_neighbors", 64)

        in_features, out_features = down_conv_nn

        # KPCONV arguments
        self.in_features = in_features
        self.out_features = out_features
        self.kp_points = kp_points

        self.pre_mlp = nn.Linear(self.in_features, self.in_features // 2)
        self._conv = PointKernel(self.kp_points, self.in_features // 2, self.in_features // 2, radius=self.radius,)
        self.post_mlp = nn.Linear(self.in_features // 2, out_features)

        self.shortcut_mlp = nn.Linear(self.in_features, self.out_features)

    def conv(self, x, pos, edge_index, batch):
        row, col = edge_index
        x_side = self.pre_mlp(x)
        x_side = self._conv(x_side, pos, edge_index)
        x_side = self.post_mlp(x_side)

        x_shortcut = self.shortcut_mlp(x)
        x_shortcut = torch.index_select(x_shortcut, 0, row)
        x_shortcut = scatter_("add", x_shortcut, col)

        return x_side + x_shortcut


####################### BUILT WITH BaseConvolutionUp ############################


class SimpleUpsampleKPConv(BaseConvolutionUp):
    def __init__(
        self, ratio=None, radius=None, up_conv_nn=None, mlp_nn=None, kp_points=16, nb_feature=0, *args, **kwargs
    ):
        super(SimpleUpsampleKPConv, self).__init__(RadiusNeighbourFinder(radius), *args, **kwargs)

        in_features, out_features = up_conv_nn

        # KPCONV arguments
        self.radius = radius
        self.in_features = in_features
        self.out_features = out_features
        self.kp_points = kp_points

        self._conv = PointKernel(self.kp_points, self.in_features, self.out_features, radius=self.radius)

        self.nn = MLP(mlp_nn, activation=LeakyReLU(0.2))

    def conv(self, x, pos, pos_skip, batch, batch_skip, edge_index):
        return self._conv(x, (pos, pos_skip), edge_index)


class ResidualUpsampleBKPConv(BaseConvolutionUp):
    def __init__(
        self, ratio=None, radius=None, up_conv_nn=None, mlp_nn=None, kp_points=16, nb_feature=0, *args, **kwargs
    ):
        super(ResidualUpsampleBKPConv, self).__init__(RadiusNeighbourFinder(radius))

        self.ratio = ratio
        self.radius = radius
        self.max_num_neighbors = kwargs.get("max_num_neighbors", 64)

        in_features, out_features = up_conv_nn

        # KPCONV arguments
        self.in_features = in_features
        self.out_features = out_features
        self.kp_points = kp_points

        self.pre_mlp = nn.Linear(self.in_features, self.in_features // 4)
        self._conv = PointKernel(self.kp_points, self.in_features // 4, self.in_features // 4, radius=self.radius,)
        self.post_mlp = nn.Linear(self.in_features // 4, out_features)

        self.shortcut_mlp = nn.Linear(self.in_features, self.out_features)

        self.nn = MLP(mlp_nn, activation=LeakyReLU(0.2))

    def conv(self, x, pos, pos_skip, batch, batch_skip, edge_index):
        row, col = edge_index
        x_side = self.pre_mlp(x)
        x_side = self._conv(x_side, (pos, pos_skip), edge_index)
        x_side = self.post_mlp(x_side)

        x_shortcut = self.shortcut_mlp(x)
        x_shortcut = torch.index_select(x_shortcut, 0, row)
        x_shortcut = scatter_("add", x_shortcut, col)

        return x_side + x_shortcut


def max_pool(features, pools):

    if pools.shape[1] > 2:
        x = torch.cat([features, torch.min(features, axis=0).values.view(1, -1)], axis=0)
        pool_features = x[pools]
        return torch.max(pool_features, axis=1).values
    else:
        row, col = pools.t()
        pool_features, _ = scatter_max(features[col], row, dim=0)
