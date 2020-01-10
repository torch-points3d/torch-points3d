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
from .convolution_ops import KPConv_ops, KPConv_deform_ops
from .utils import weight_variable
from .kernels import PointKernel, LightDeformablePointKernel
from .kernel_utils import kernel_point_optimization_debug
from models.core_sampling_and_search import RadiusNeighbourFinder, FPSSampler
from models.core_modules import *
from models.unet_base import BaseFactory


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


####################### BUILT WITH PARTIAL DENSE FORMAT ############################


class BaseKPConvPartialDense(BaseConvolutionDown):
    def __init__(
        self,
        ratio=None,
        radius=None,
        down_conv_nn=None,
        kp_points=16,
        nb_feature=0,
        is_strided=False,
        KP_EXTENT=None,
        DENSITY_PARAMETER=None,
        *args,
        **kwargs
    ):
        super(BaseKPConvPartialDense, self).__init__(
            FPSSampler(ratio), RadiusNeighbourFinder(radius, conv_type=kwargs.get("conv_type")), *args, **kwargs
        )

        self.ratio = ratio
        self.radius = radius
        self.is_strided = is_strided

        if len(down_conv_nn) == 2:
            in_features, out_features = down_conv_nn
            intermediate_features = None

        elif len(down_conv_nn) == 3:
            in_features, intermediate_features, out_features = down_conv_nn

        else:
            raise NotImplementedError

        # KPCONV arguments
        self.in_features = in_features
        self.out_features = out_features
        self.intermediate_features = intermediate_features
        self.kp_points = kp_points

        # Dataset ~ Model parameters
        self.KP_EXTENT = KP_EXTENT
        self.DENSITY_PARAMETER = DENSITY_PARAMETER

        # PARAMTERS IMPORTANT FOR SHADOWING
        self.shadow_features_fill = 0.0
        self.shadow_points_fill_ = float(10e6)


class KPConvPartialDense(BaseKPConvPartialDense):
    def __init__(self, *args, **kwargs):
        super(KPConvPartialDense, self).__init__(*args, **kwargs)

        self._conv = PointKernelPartialDense(
            self.kp_points,
            self.in_features,
            self.out_features,
            radius=self.radius,
            is_strided=self.is_strided,
            KP_EXTENT=self.KP_EXTENT,
            DENSITY_PARAMETER=self.DENSITY_PARAMETER,
        )
        self.activation = kwargs.get("act", nn.LeakyReLU(0.2))

    def conv(self, input, pos, input_neighbour, pos_neighbour, idx_neighbour, idx_sampler):
        return self._conv(input_neighbour, pos_neighbour, idx_sampler)


class ResnetPartialDense(BaseKPConvPartialDense):
    def __init__(self, *args, **kwargs):
        super(ResnetPartialDense, self).__init__(*args, **kwargs)

        self._kp_conv0 = PointKernelPartialDense(
            self.kp_points,
            self.in_features,
            self.intermediate_features,
            radius=self.radius,
            is_strided=False,
            KP_EXTENT=self.KP_EXTENT,
            DENSITY_PARAMETER=self.DENSITY_PARAMETER,
        )

        self._kp_conv1 = PointKernelPartialDense(
            self.kp_points,
            self.intermediate_features,
            self.out_features,
            radius=self.radius,
            is_strided=self.is_strided,
            KP_EXTENT=self.KP_EXTENT,
            DENSITY_PARAMETER=self.DENSITY_PARAMETER,
        )

        if self.out_features != self.intermediate_features:
            self.shortcut_op = UnaryConv(self.intermediate_features, self.out_features)
        else:
            self.shortcut_op = torch.nn.Identity()

    def conv(self, input, pos, input_neighbour, pos_centered_neighbour, idx_neighbour, idx_sampler):

        x = self.kp_conv0(input, idx_neighbour, pos_centered_neighbour, idx_sampler=None)

        x = self.kp_conv1(x, idx_neighbour, pos_centered_neighbour, idx_sampler)

        if self.is_strided:
            x = x[idx_neighbour][idx_sampler].max(-1)
        x = x + self.shortcut_op(input)

        return x


class ResnetBottleNeckPartialDense(BaseKPConvPartialDense):
    def __init__(self, *args, **kwargs):
        super(ResnetBottleNeckPartialDense, self).__init__(*args, **kwargs)

        self._kp_conv0 = PointKernelPartialDense(
            self.kp_points,
            self.intermediate_features,
            self.intermediate_features,
            radius=self.radius,
            is_strided=False,
            KP_EXTENT=self.KP_EXTENT,
            DENSITY_PARAMETER=self.DENSITY_PARAMETER,
        )

        self._kp_conv1 = PointKernelPartialDense(
            self.kp_points,
            self.intermediate_features,
            self.intermediate_features,
            radius=self.radius,
            is_strided=self.is_strided,
            KP_EXTENT=self.KP_EXTENT,
            DENSITY_PARAMETER=self.DENSITY_PARAMETER,
        )

        self.uconv_0 = UnaryConv(self.in_features, self.intermediate_features)

        if self.out_features != self.intermediate_features:
            self.shortcut_op = UnaryConv(self.intermediate_features, self.out_features)
        else:
            self.shortcut_op = torch.nn.Identity()

    def conv(self, input, pos, input_neighbour, pos_centered_neighbour, idx_neighbour, idx_sampler):

        x = self.uconv_0(input)
        x = self._kp_conv0(x, idx_neighbour, pos_centered_neighbour, idx_sampler=None)
        x = self._kp_conv1(x, idx_neighbour, pos_centered_neighbour, idx_sampler=idx_sampler)
        if self.is_strided:
            x = input_neighbour[idx_sampler].max(-1)
        x = x + self.shortcut_op(input)

        return x


####################### BUILT WITH BaseConvolutionDown ############################


class LightDeformableKPConv(BaseConvolutionDown):
    def __init__(self, ratio=None, radius=None, down_conv_nn=None, num_points=16, nb_feature=0, *args, **kwargs):
        super(LightDeformableKPConv, self).__init__(FPSSampler(ratio), RadiusNeighbourFinder(radius), *args, **kwargs)

        self.ratio = ratio
        self.radius = radius

        in_features, out_features = down_conv_nn

        # KPCONV arguments
        self.in_features = in_features
        self.out_features = out_features
        self.num_points = num_points

        self._conv = LightDeformablePointKernel(
            self.num_points, self.in_features, self.out_features, radius=self.radius
        )

    def conv(self, x, pos, edge_index, batch):
        return self._conv(x, pos, edge_index)


class KPConv(BaseConvolutionDown):
    def __init__(self, ratio=None, radius=None, down_conv_nn=None, num_points=16, nb_feature=0, *args, **kwargs):
        super(KPConv, self).__init__(FPSSampler(ratio), RadiusNeighbourFinder(radius), *args, **kwargs)

        self.ratio = ratio
        self.radius = radius

        in_features, out_features = down_conv_nn

        # KPCONV arguments
        self.in_features = in_features
        self.out_features = out_features
        self.num_points = num_points

        self._conv = PointKernel(self.num_points, self.in_features, self.out_features, radius=self.radius)

    def conv(self, x, pos, edge_index, batch):
        return self._conv(x, pos, edge_index)


class ResidualBKPConv(BaseConvolutionDown):
    def __init__(self, ratio=None, radius=None, down_conv_nn=None, num_points=16, nb_feature=0, *args, **kwargs):
        super(ResidualBKPConv, self).__init__(FPSSampler(ratio), RadiusNeighbourFinder(radius), *args, **kwargs)

        self.ratio = ratio
        self.radius = radius
        self.max_num_neighbors = kwargs.get("max_num_neighbors", 64)

        in_features, out_features = down_conv_nn

        # KPCONV arguments
        self.in_features = in_features
        self.out_features = out_features
        self.num_points = num_points

        self.pre_mlp = nn.Linear(self.in_features, self.in_features // 2)
        self._conv = PointKernel(self.num_points, self.in_features // 2, self.in_features // 2, radius=self.radius,)
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
        self, ratio=None, radius=None, up_conv_nn=None, mlp_nn=None, num_points=16, nb_feature=0, *args, **kwargs
    ):
        super(SimpleUpsampleKPConv, self).__init__(RadiusNeighbourFinder(radius), *args, **kwargs)

        in_features, out_features = up_conv_nn

        # KPCONV arguments
        self.radius = radius
        self.in_features = in_features
        self.out_features = out_features
        self.num_points = num_points

        self._conv = PointKernel(self.num_points, self.in_features, self.out_features, radius=self.radius)

        self.nn = MLP(mlp_nn, activation=LeakyReLU(0.2))

    def conv(self, x, pos, pos_skip, batch, batch_skip, edge_index):
        return self._conv(x, (pos, pos_skip), edge_index)


class ResidualUpsampleBKPConv(BaseConvolutionUp):
    def __init__(
        self, ratio=None, radius=None, up_conv_nn=None, mlp_nn=None, num_points=16, nb_feature=0, *args, **kwargs
    ):
        super(ResidualUpsampleBKPConv, self).__init__(RadiusNeighbourFinder(radius))

        self.ratio = ratio
        self.radius = radius
        self.max_num_neighbors = kwargs.get("max_num_neighbors", 64)

        in_features, out_features = up_conv_nn

        # KPCONV arguments
        self.in_features = in_features
        self.out_features = out_features
        self.num_points = num_points

        self.pre_mlp = nn.Linear(self.in_features, self.in_features // 4)
        self._conv = PointKernel(self.num_points, self.in_features // 4, self.in_features // 4, radius=self.radius,)
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


# Kernel Point Convolution in Pytorch
# Adaption from https://github.com/humanpose1/KPConvTorch/blob/master/models/layers.py


class KPConvLayer(torch.nn.Module):
    """
    apply the kernel point convolution on a point cloud
    NB : it is the original version of KPConv, it is not the message passing version
    attributes:
    layer_ind (int): index of the layer
    radius: radius of the kernel
    num_inputs : dimension of the input feature
    num_outputs : dimension of the output feature
    config : YACS class that contains all the important constants
    and hyperparameters
    """

    def __init__(self, radius, num_inputs, num_outputs, config):
        super(KPConvLayer, self).__init__()
        self.radius = radius
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.config = config
        self.extent = self.config.NETWORK.KP_EXTENT * self.radius / self.config.NETWORK.DENSITY_PARAMETER

        # Initial kernel extent for this layer
        K_radius = 1.5 * self.extent
        K_points_numpy = create_kernel_points(
            K_radius,
            self.config.NETWORK.NUM_KERNEL_POINTS,
            num_kernels=1,
            dimension=self.config.INPUT.POINTS_DIM,
            fixed=self.config.NETWORK.FIXED_KERNEL_POINTS,
        )

        self.K_points = Parameter(
            torch.from_numpy(
                K_points_numpy.reshape((self.config.NETWORK.NUM_KERNEL_POINTS, self.config.INPUT.POINTS_DIM,))
            ).to(torch.float),
            requires_grad=False,
        )

        self.weight = Parameter(
            weight_variable([self.config.NETWORK.NUM_KERNEL_POINTS, self.num_inputs, self.num_outputs,])
        )

    def forward(self, pos, neighbors, x):
        """
        - pos is a tuple containing:
          - query_points(torch Tensor): query of size N x 3
          - support_points(torch Tensor): support points of size N0 x 3
        - neighbors(torch Tensor): neighbors of size N0 x M
        - features : feature of size N x d (d is the number of inputs)
        """
        support_points, query_points = pos
        new_feat = KPConv_ops(
            query_points,
            support_points,
            neighbors,
            x,
            self.K_points,
            self.weight,
            self.extent,
            self.config.NETWORK.KP_INFLUENCE,
            self.config.NETWORK.CONVOLUTION_MODE,
        )
        return new_feat


class DeformableKPConvLayer(torch.nn.Module):
    def __init__(self, radius, num_inputs, num_outputs, config, version=0, modulated=False):
        """
        it doesn't work yet :
        """
        super(DeformableKPConvLayer, self).__init__()
        self.radius = radius
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.config = config
        self.extent = self.config.NETWORK.KP_EXTENT * self.radius / self.config.NETWORK.DENSITY_PARAMETER
        self.version = version
        self.modulated = modulated

        # Initial kernel extent for this layer
        K_radius = 1.5 * self.extent
        K_points_numpy = create_kernel_points(
            K_radius,
            self.config.NETWORK.NUM_KERNEL_POINTS,
            num_kernels=1,
            dimension=self.config.INPUT.POINTS_DIM,
            fixed=self.config.NETWORK.FIXED_KERNEL_POINTS,
        )

        self.K_points = Parameter(
            torch.from_numpy(
                K_points_numpy.reshape((self.config.NETWORK.NUM_KERNEL_POINTS, self.config.INPUT.POINTS_DIM,))
            ).to(torch.float),
            requires_grad=False,
        )

        # Parameter of the deformable convolution
        self.weight = Parameter(
            weight_variable([self.config.NETWORK.NUM_KERNEL_POINTS, self.num_inputs, self.num_outputs,])
        )
        if self.modulated:
            offset_dim = (self.config.INPUT.POINTS_DIM + 1) * (self.config.NETWORK.NUM_KERNEL_POINTS - 1)
        else:
            offset_dim = (self.config.INPUT.POINTS_DIM) * (self.config.NETWORK.NUM_KERNEL_POINTS - 1)

        if self.version == 0:
            # kp conv to estimate the offset
            self.deformable_weight = Parameter(
                weight_variable([self.config.NETWORK.NUM_KERNEL_POINTS, self.num_inputs, offset_dim])
            )
        elif self.version == 1:
            # MLP to estimate the offset
            self.deformable_weight = Parameter(weight_variable([self.num_inputs, offset_dim]))
        self.bias = torch.nn.Parameter(torch.zeros(offset_dim, dtype=torch.float32), requires_grad=True)

    def forward(self, query_points, support_points, neighbors, features):

        points_dim = self.config.INPUT.POINTS_DIM
        num_kpoints = self.config.NETWORK.NUM_KERNEL_POINTS
        if self.version == 0:
            features0 = (
                KPConv_ops(
                    query_points,
                    support_points,
                    neighbors,
                    features,
                    self.K_points,
                    self.deformable_weight,
                    self.extent,
                    self.config.NETWORK.KP_INFLUENCE,
                    self.config.NETWORK.CONVOLUTION_MODE,
                )
                + self.bias
            )

        if self.modulated:
            # Get offset (in normalized scale) from features
            offsets = features0[:, : points_dim * (num_kpoints - 1)]
            offsets = offsets.reshape([-1, (num_kpoints - 1), points_dim])

            # Get modulations
            modulations = 2 * torch.sigmoid(features0[:, points_dim * (num_kpoints - 1) :])

            #  No offset for the first Kernel points
            if self.version == 1:
                offsets = torch.cat([torch.zeros_like(offsets[:, :1, :]), offsets], axis=1)
                modulations = torch.cat([torch.zeros_like(modulations[:, :1]), modulations], axis=1)
        else:
            # Get offset (in normalized scale) from features
            offsets = features0.reshape([-1, (num_kpoints - 1), points_dim])
            # No offset for the first Kernel points
            offsets = torch.cat([torch.zeros_like(offsets[:, :1, :]), offsets], axis=1)

            # No modulations
            modulations = None

        # Rescale offset for this layer
        offsets *= self.config.NETWORK.KP_EXTENT
        feat, sq_distances, _ = KPConv_deform_ops(
            query_points,
            support_points,
            neighbors,
            features,
            self.K_points,
            offsets,
            modulations,
            self.weight,
            self.extent,
            self.config.NETWORK.KP_INFLUENCE,
            self.config.NETWORK.CONVOLUTION_MODE,
        )
        self.sq_distances = torch.nn.Parameter(sq_distances)
        return feat


class UnaryConv(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        """
        1x1 convolution on point cloud (we can even call it a mini pointnet)
        """
        super(UnaryConv, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.weight = Parameter(weight_variable([self.num_inputs, self.num_outputs]))

    def forward(self, features):
        """
        features(Torch Tensor): size N x d d is the size of inputs
        """
        return torch.matmul(features, self.weight)

    def __repr__(self):
        return "UnaryConv({}, {})".format(self.num_inputs, self.num_outputs)


def max_pool(features, pools):

    if pools.shape[1] > 2:
        x = torch.cat([features, torch.min(features, axis=0).values.view(1, -1)], axis=0)
        pool_features = x[pools]
        return torch.max(pool_features, axis=1).values
    else:
        row, col = pools.t()
        pool_features, _ = scatter_max(features[col], row, dim=0)
