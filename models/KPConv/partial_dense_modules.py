import torch
from torch import nn

# Kernel Point Convolution in Pytorch
# Adaption from https://github.com/humanpose1/KPConvTorch/blob/master/models/layers.py
from .kernel_utils import kernel_point_optimization_debug
from models.core_sampling_and_search import RadiusNeighbourFinder, FPSSampler, GridSampler
from models.partial_dense_modules import BasePartialDenseConvolutionDown
from .kernels import PointKernelPartialDense
from models.core_modules import UnaryConv

####################### BUILT WITH PARTIAL DENSE FORMAT ############################


class BaseKPConvPartialDense(BasePartialDenseConvolutionDown):
    def __init__(
        self,
        ratio=None,
        radius=None,
        down_conv_nn=None,
        kp_points=16,
        nb_feature=0,
        is_strided=True,
        kp_extent=1,
        density_parameter=1,
        *args,
        **kwargs
    ):
        subsampling_param = (kwargs.get("index") + 1) * kwargs.get("first_subsampling", 0.01)
        super(BaseKPConvPartialDense, self).__init__(
            FPSSampler(ratio),
            RadiusNeighbourFinder(
                radius, max_num_neighbors=kwargs.get("max_num_neighbors", 64), conv_type=self.CONV_TYPE
            ),
            *args,
            **kwargs
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
        self.kp_extent = kp_extent
        self.density_parameter = density_parameter

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
            kp_extent=self.kp_extent,
            density_parameter=self.density_parameter,
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
            kp_extent=self.kp_extent,
            density_parameter=self.density_parameter,
        )

        self._kp_conv1 = PointKernelPartialDense(
            self.kp_points,
            self.intermediate_features,
            self.out_features,
            radius=self.radius,
            is_strided=self.is_strided,
            kp_extent=self.kp_extent,
            density_parameter=self.density_parameter,
        )

        if self.out_features != self.intermediate_features:
            self.shortcut_op = UnaryConv(self.intermediate_features, self.out_features)
        else:
            self.shortcut_op = torch.nn.Identity()

    def conv(self, input, pos, input_neighbour, pos_centered_neighbour, idx_neighbour, idx_sampler):

        x = self._kp_conv0(input, idx_neighbour, pos_centered_neighbour, idx_sampler=None)
        x = self._kp_conv1(x, idx_neighbour, pos_centered_neighbour, idx_sampler=idx_sampler)

        if self.is_strided:
            input = input_neighbour[idx_sampler].max(1)[0]
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
            kp_extent=self.kp_extent,
            density_parameter=self.density_parameter,
        )

        self._kp_conv1 = PointKernelPartialDense(
            self.kp_points,
            self.intermediate_features,
            self.out_features,
            radius=self.radius,
            is_strided=self.is_strided,
            kp_extent=self.kp_extent,
            density_parameter=self.density_parameter,
        )

        self.uconv_0 = UnaryConv(self.in_features, self.intermediate_features)

        if self.out_features != self.intermediate_features:
            self.shortcut_op = UnaryConv(self.in_features, self.out_features)
        else:
            self.shortcut_op = torch.nn.Identity()

        self.norm = nn.BatchNorm1d(self.out_features)
        self.act = nn.LeakyReLU(0.2)

    def conv(self, input, pos, input_neighbour, pos_centered_neighbour, idx_neighbour, idx_sampler):

        x = self.uconv_0(input)
        x = self._kp_conv0(x, idx_neighbour, pos_centered_neighbour, idx_sampler=None)
        x = self._kp_conv1(x, idx_neighbour, pos_centered_neighbour, idx_sampler=idx_sampler)

        if self.is_strided:
            input = input_neighbour[idx_sampler].max(1)[0]
        x = x + self.shortcut_op(input)
        return self.act(self.norm(x))
