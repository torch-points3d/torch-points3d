# Kernel Point Convolution in Pytorch
# Adaption from https://github.com/humanpose1/KPConvTorch/blob/master/models/layers.py
from .kernel_utils import kernel_point_optimization_debug
from src.core.spatial_ops import FPSSampler, RadiusNeighbourFinder
from src.core.base_conv.partial_dense import *
from src.core.common_modules import UnaryConv

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
