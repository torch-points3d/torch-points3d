import numpy as np
from enum import Enum
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import math
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing
from models.base_model import BaseInternalLossModule

from .kernel_utils import kernel_point_optimization_debug
from .convolution_ops import *


def KPconv_op(self, x_j, pos_i, pos_j):
    if x_j is None:
        x_j = pos_j

    # Center every neighborhood [SUM n_neighbors(n_points), dim]
    neighbors = pos_j - pos_i

    # Number of points
    n_points = neighbors.shape[0]

    # Get points kernels
    K_points = self.kernel

    # Get all difference matrices [SUM n_neighbors(n_points), n_kpoints, dim]
    neighbors = neighbors.unsqueeze(1)

    differences = neighbors - K_points.float().view((-1, 3)).unsqueeze(0)
    sq_distances = (differences ** 2).sum(-1)

    # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
    if self.KP_influence == "constant":
        # Every point get an influence of 1.
        all_weights = torch.ones_like(sq_distances)

    elif self.KP_influence == "linear":
        # Influence decrease linearly with the distance, and get to zero when d = kp_extent.
        all_weights = 1.0 - (sq_distances / (self.kp_extent ** 2))
        all_weights[all_weights < 0] = 0.0
    else:
        raise ValueError("Unknown influence function type (config.KP_influence)")

    neighbors_1nn = torch.argmin(sq_distances, dim=-1)
    weights = all_weights.gather(1, neighbors_1nn.unsqueeze(-1))

    K_weights = self.kernel_weight
    K_weights = torch.index_select(K_weights, 0, neighbors_1nn.view(-1)).view(
        (n_points, self.in_features, self.out_features)
    )

    # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
    features = x_j

    # Apply distance weights [n_points, n_kpoints, in_fdim]
    weighted_features = torch.einsum("nb, nc -> nc", weights, features)

    # Apply network weights [n_kpoints, n_points, out_fdim]
    out_features = torch.einsum("na, nac -> nc", weighted_features, K_weights)

    return out_features, neighbors


class PointKernel(MessagePassing):
    """
    Implements KPConv: Flexible and Deformable Convolution for Point Clouds from
    https://arxiv.org/abs/1904.08889
    """

    def __init__(
        self,
        num_points,
        in_features,
        out_features,
        radius=1,
        kernel_dim=3,
        fixed="center",
        ratio=1,
        KP_influence="linear",
    ):
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
        self.kp_extent = radius / 1.5

        # Point position in kernel_dim
        self.kernel = Parameter(torch.Tensor(1, num_points, kernel_dim))

        # Associated weights
        self.kernel_weight = Parameter(torch.Tensor(num_points, in_features, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.kernel_weight, a=math.sqrt(5))

        # Init the kernel using attrative + repulsion forces
        kernel, _ = kernel_point_optimization_debug(
            self.radius,
            self.num_points,
            num_kernels=1,
            dimension=self.kernel_dim,
            fixed=self.fixed,
            ratio=self.ratio,
            verbose=False,
        )
        self.kernel.data = torch.from_numpy(kernel)

    def forward(self, x, pos, edge_index):
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_j, pos_i, pos_j):
        out_features, _ = KPconv_op(self, x_j, pos_i, pos_j)
        return out_features

    def update(self, aggr_out, pos):
        return aggr_out

    def __repr__(self):
        # PointKernel parameters
        return "PointKernel({}, {}, {}, {}, {})".format(
            self.in_features, self.out_features, self.num_points, self.radius, self.KP_influence,
        )


class LossKPConvEnum(Enum):
    PERMISSIVE = 0
    FITTING = 1
    REPULSION = 2


def repulsion_loss(deformed_kpoints):
    pass


def permissive_loss(deformed_kpoints, radius):
    """This loss is responsible to penalize deformed_kpoints to
    move outside from the radius defined for the convolution
    """
    norm_deformed_normalized = F.normalize(deformed_kpoints) / float(radius)
    return torch.mean(norm_deformed_normalized[norm_deformed_normalized > 1.0])


# Implements the Light Deformable KPConv
# https://github.com/HuguesTHOMAS/KPConv/blob/master/kernels/convolution_ops.py#L503


class LightDeformablePointKernel(MessagePassing, BaseInternalLossModule):

    """
    Implements KPConv: Flexible and Deformable Convolution for Point Clouds from
    https://arxiv.org/abs/1904.08889
    """

    def __init__(
        self,
        num_points,
        in_features,
        out_features,
        radius=1,
        kernel_dim=3,
        fixed="center",
        ratio=1,
        KP_influence="square",
        modulated=False,
    ):
        super(LightDeformablePointKernel, self).__init__()
        # PointKernel parameters
        self.in_features = in_features
        self.out_features = out_features
        self.num_points = num_points
        self.radius = radius
        self.kernel_dim = kernel_dim
        self.fixed = fixed
        self.ratio = ratio
        self.KP_influence = KP_influence
        self.modulated = modulated

        # Radius of the initial positions of the kernel points
        self.kp_extent = radius / 1.5

        # Point position in kernel_dim
        self.kernel = Parameter(torch.Tensor(1, num_points, kernel_dim))
        # Associated weights
        self.kernel_weight = Parameter(torch.Tensor(num_points, in_features, out_features))

        # Linear Kernel to learn offset.
        if modulated:
            offset_dim = (kernel_dim + 1) * (num_points - 1)
        else:
            offset_dim = kernel_dim * (num_points)

        self.offset_mlp = nn.Linear(in_features, offset_dim, bias=False)

        self.reset_parameters()

        self.internal_losses = {}

    def reset_parameters(self):
        init.kaiming_uniform_(self.kernel_weight, a=math.sqrt(5))

        self.offset_mlp.reset_parameters()

        # Init the kernel using attrative + repulsion forces
        kernel, _ = kernel_point_optimization_debug(
            self.radius,
            self.num_points,
            num_kernels=1,
            dimension=self.kernel_dim,
            fixed=self.fixed,
            ratio=self.ratio,
            verbose=False,
        )
        self.kernel.data = torch.from_numpy(kernel)

    def forward(self, x, pos, edge_index):
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_j, pos_i, pos_j):
        if x_j is None:
            x_j = pos_j

        # Get offsets from points
        offsets = self.offset_mlp(x_j)

        # Reshape offsets to shape [SUM n_neighbors(n_points), n_kpoints, kp_dim]
        offsets = offsets.view((-1, self.num_points, self.kernel_dim))

        # Rescale offset for this layer
        offsets *= self.kp_extent

        # Center every neighborhood [SUM n_neighbors(n_points), dim]
        neighbors = pos_j - pos_i

        # Number of points
        n_points = neighbors.shape[0]

        # Get points kernels and add offsets
        K_points = self.kernel
        K_points = K_points.float().view((-1, 3)).unsqueeze(0)
        K_points_deformed = K_points + offsets
        self.internal_losses["permissive_loss"] = permissive_loss(K_points_deformed, self.radius)

        # Get all difference matrices [SUM n_neighbors(n_points), n_kpoints, dim]
        neighbors = neighbors.unsqueeze(1)

        differences = neighbors - K_points_deformed

        sq_distances = (differences ** 2).sum(-1)

        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        if self.KP_influence == "constant":
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)

        elif self.KP_influence == "linear":
            # Influence decrease linearly with the distance, and get to zero when d = kp_extent.
            all_weights = 1.0 - (torch.sqrt(sq_distances) / (self.kp_extent))
            all_weights[all_weights < 0] = 0.0

        elif self.KP_influence == "square":
            # Influence decrease linearly with the distance, and get to zero when d = kp_extent.
            all_weights = 1.0 - (sq_distances / (self.kp_extent ** 2))
            all_weights[all_weights < 0] = 0.0

        else:
            raise ValueError("Unknown influence function type (config.KP_influence)")

        neighbors_1nn = torch.argmin(sq_distances, dim=-1)

        # Fitting Loss
        sq_distances_min = sq_distances.gather(1, neighbors_1nn.unsqueeze(-1))
        sq_distances_min /= self.radius ** 2  # To be independant of the layer
        self.internal_losses["fitting_loss"] = torch.mean(sq_distances_min)

        weights = all_weights.gather(1, neighbors_1nn.unsqueeze(-1))

        K_weights = self.kernel_weight
        K_weights = torch.index_select(K_weights, 0, neighbors_1nn.view(-1)).view(
            (n_points, self.in_features, self.out_features)
        )

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        features = x_j

        # Apply distance weights [n_points, n_kpoints, in_fdim]
        weighted_features = torch.einsum("nb, nc -> nc", weights, features)

        # Apply network weights [n_kpoints, n_points, out_fdim]
        out_features = torch.einsum("na, nac -> nc", weighted_features, K_weights)

        return out_features

    def get_internal_losses(self):
        return self.internal_losses

    def update(self, aggr_out, pos):
        return aggr_out

    def __repr__(self):
        # PointKernel parameters
        return "PointKernel({}, {}, {}, {}, {})".format(
            self.in_features, self.out_features, self.num_points, self.radius, self.KP_influence
        )


####################### BUILT WITH PARTIAL DENSE FORMAT ############################


class PointKernelPartialDense(nn.Module):
    """
    Implements KPConv: Flexible and Deformable Convolution for Point Clouds from
    https://arxiv.org/abs/1904.08889
    """

    def __init__(
        self,
        num_points,
        in_features,
        out_features,
        radius=1,
        kernel_dim=3,
        fixed="center",
        ratio=1,
        KP_influence="linear",
        aggregation_mode="closest",
        is_strided=True,
        shadow_features_fill=0.0,
        norm=nn.BatchNorm1d,
        act=nn.LeakyReLU,
        kp_extent=None,
        density_parameter=None,
    ):
        super(PointKernelPartialDense, self).__init__()
        # PointKernel parameters
        self.in_features = in_features
        self.out_features = out_features
        self.num_points = num_points
        self.radius = radius
        self.kernel_dim = kernel_dim
        self.fixed = fixed
        self.ratio = ratio
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.is_strided = is_strided
        self.norm = norm(out_features) if norm is not None else None
        self.act = act()

        # Position of the fill for shadow points
        self.shadow_features_fill = shadow_features_fill

        # Radius of the initial positions of the kernel points
        self.extent = kp_extent * self.radius / density_parameter

        # Initial kernel extent for this layer
        self.K_radius = 1.5 * self.extent

        self.kp_extent = radius / 1.5

        # Point position in kernel_dim
        self.kernel = Parameter(torch.Tensor(1, num_points, kernel_dim)).float()

        # Associated weights
        self.kernel_weight = Parameter(torch.Tensor(num_points, in_features, out_features)).float()

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.kernel_weight, a=math.sqrt(5))

        # Init the kernel using attrative + repulsion forces
        kernel, _ = kernel_point_optimization_debug(
            self.K_radius,
            self.num_points,
            num_kernels=1,
            dimension=self.kernel_dim,
            fixed=self.fixed,
            ratio=self.ratio,
            verbose=False,
        )
        self.kernel.data = torch.from_numpy(kernel).float()

    def forward(self, x, idx_neighbour, pos_centered_neighbour, idx_sampler=None):
        if idx_sampler is None and self.is_strided:
            raise Exception("This convolution needs to be provided idx_sampler as it is defined as strided")

        features = KPConv_ops_partial(
            x,
            idx_neighbour,
            pos_centered_neighbour,
            idx_sampler,
            self.kernel,
            self.kernel_weight,
            self.extent,
            self.KP_influence,
            self.aggregation_mode,
        )

        if not self.is_strided:
            shadow_features = torch.full((1,) + features.shape[1:], self.shadow_features_fill).to(x.device)
            features = torch.cat([features, shadow_features], dim=0)

        if self.norm:
            return self.act(self.norm(features))
        return self.act(features)

    def __repr__(self):
        # PointKernel parameters
        return "PointKernel({}, {}, {}, {}, {})".format(
            self.in_features, self.out_features, self.num_points, self.radius, self.KP_influence
        )
