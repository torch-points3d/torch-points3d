from enum import Enum
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import math
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing

from .kernel_utils import kernel_point_optimization_debug, load_kernels
from .convolution_ops import *
from src.models.base_model import BaseInternalLossModule


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


def fitting_loss(sq_distance, radius):
    kpmin = sq_distance.min(dim=1)
    normalised_kpmin = kpmin / (radius ** 2)
    return torch.mean(normalised_kpmin)


def repulsion_loss(deformed_kpoints, radius):
    deformed_kpoints / float(radius)
    n_points = deformed_kpoints.shape[0]
    repulsive_loss = 0
    for i in range(n_points):
        with torch.no_grad():
            other_points = torch.cat([deformed_kpoints[:, :i, :], deformed_kpoints[:, i + 1 :, :]], dim=1)
        distances = torch.sqrt(torch.sum((other_points - deformed_kpoints[:, i : i + 1, :]) ** 2, dim=-1))
        repulsion_force = torch.sum(torch.square(torch.max(0, 1.5 - distances)), dim=1)
        repulsive_loss += torch.mean(repulsion_force)
    return repulsive_loss


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


class KPConvLayer(torch.nn.Module):
    """
    apply the kernel point convolution on a point cloud
    NB : it is the original version of KPConv, it is not the message passing version
    attributes:
    num_inputs : dimension of the input feature
    num_outputs : dimension of the output feature
    point_influence: influence distance of a single point (sigma * grid_size)
    n_kernel_points=15
    fixed="center"
    KP_influence="linear"
    aggregation_mode="sum"
    dimension=3
    """

    _INFLUENCE_TO_RADIUS = 1.5

    def __init__(
        self,
        num_inputs,
        num_outputs,
        point_influence,
        n_kernel_points=15,
        fixed="center",
        KP_influence="linear",
        aggregation_mode="sum",
        dimension=3,
    ):
        super(KPConvLayer, self).__init__()
        self.kernel_radius = self._INFLUENCE_TO_RADIUS * point_influence
        self.point_influence = point_influence
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.KP_influence = KP_influence
        self.n_kernel_points = n_kernel_points
        self.aggregation_mode = aggregation_mode

        # Initial kernel extent for this layer
        K_points_numpy = load_kernels(
            self.kernel_radius, n_kernel_points, num_kernels=1, dimension=dimension, fixed=fixed,
        )

        self.K_points = Parameter(
            torch.from_numpy(K_points_numpy.reshape((n_kernel_points, dimension))).to(torch.float), requires_grad=False,
        )

        weights = torch.empty([n_kernel_points, num_inputs, num_outputs], dtype=torch.float)
        torch.nn.init.xavier_normal_(weights)
        self.weight = Parameter(weights)

    def forward(self, query_points, support_points, neighbors, x):
        """
        - query_points(torch Tensor): query of size N x 3
        - support_points(torch Tensor): support points of size N0 x 3
        - neighbors(torch Tensor): neighbors of size N x M
        - features : feature of size N0 x d (d is the number of inputs)
        """
        new_feat = KPConv_ops(
            query_points,
            support_points,
            neighbors,
            x,
            self.K_points,
            self.weight,
            self.point_influence,
            self.KP_influence,
            self.aggregation_mode,
        )
        return new_feat

    def __repr__(self):
        return "KPConvLayer(InF: %i, OutF: %i, kernel_pts: %i, radius: %.2f, KP_influence: %s)" % (
            self.num_inputs,
            self.num_outputs,
            self.n_kernel_points,
            self.kernel_radius,
            self.KP_influence,
        )


class KPConvDeformableLayer(torch.nn.Module, BaseInternalLossModule):
    """
    apply the deformable kernel point convolution on a point cloud
    NB : it is the original version of KPConv, it is not the message passing version
    attributes:
    num_inputs : dimension of the input feature
    num_outputs : dimension of the output feature
    point_influence: influence distance of a single point (sigma * grid_size)
    n_kernel_points=15
    fixed="center"
    KP_influence="linear"
    aggregation_mode="sum"
    dimension=3
    modulated = False :   If deformable conv should be modulated
    """

    OFFSET_LOSS_KEY = "offset_loss"
    _INFLUENCE_TO_RADIUS = 1.5

    def __init__(
        self,
        num_inputs,
        num_outputs,
        point_influence,
        n_kernel_points=15,
        fixed="center",
        KP_influence="linear",
        aggregation_mode="sum",
        dimension=3,
        modulated=False,
        loss_mode="fitting",
        loss_decay=0.1,
    ):
        super(KPConvDeformableLayer, self).__init__()
        self.kernel_radius = self._INFLUENCE_TO_RADIUS * point_influence
        self.point_influence = point_influence
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.KP_influence = KP_influence
        self.n_kernel_points = n_kernel_points
        self.aggregation_mode = aggregation_mode
        self.modulated = modulated
        self.internal_losses = {self.OFFSET_LOSS_KEY: 0}
        self.loss_mode = loss_mode
        self.loss_decay = loss_decay

        # Initial kernel extent for this layer
        K_points_numpy = load_kernels(
            self.kernel_radius, n_kernel_points, num_kernels=1, dimension=dimension, fixed=fixed,
        )
        self.K_points = Parameter(
            torch.from_numpy(K_points_numpy.reshape((n_kernel_points, dimension))).to(torch.float), requires_grad=False,
        )

        # Create independant weight for the first convolution and a bias term as no batch normalization happen
        if modulated:
            offset_dim = (dimension + 1) * self.n_kernel_points
        else:
            offset_dim = dimension * self.n_kernel_points
        offset_weights = torch.empty([n_kernel_points, num_inputs, offset_dim], dtype=torch.float)
        torch.nn.init.xavier_normal_(offset_weights)
        self.offset_weights = Parameter(offset_weights)
        self.offset_bias = Parameter(torch.zeros(offset_dim, dtype=torch.float))

        # Main deformable weights
        weights = torch.empty([n_kernel_points, num_inputs, num_outputs], dtype=torch.float)
        torch.nn.init.xavier_normal_(weights)
        self.weight = Parameter(weights)

    def forward(self, query_points, support_points, neighbors, x):
        """
        - query_points(torch Tensor): query of size N x 3
        - support_points(torch Tensor): support points of size N0 x 3
        - neighbors(torch Tensor): neighbors of size N x M
        - features : feature of size N0 x d (d is the number of inputs)
        """
        offset_feat = (
            KPConv_ops(
                query_points,
                support_points,
                neighbors,
                x,
                self.K_points,
                self.offset_weights,
                self.point_influence,
                self.KP_influence,
                self.aggregation_mode,
            )
            + self.offset_bias
        )
        points_dim = query_points.shape[-1]
        if self.modulated:
            # Get offset (in normalized scale) from features
            offsets = offset_feat[:, : points_dim * self.n_kernel_points]
            offsets = offsets.reshape((-1, self.n_kernel_points, points_dim))

            # Get modulations
            modulations = 2 * torch.nn.functional.sigmoid(offset_feat[:, points_dim * self.n_kernel_points :])
        else:
            # Get offset (in normalized scale) from features
            offsets = offset_feat.reshape((-1, self.n_kernel_points, points_dim))
            # No modulations
            modulations = None
        offsets *= self.point_influence

        # Apply deformable kernel
        new_feat, sq_distances, K_points_deformed = KPConv_deform_ops(
            query_points,
            support_points,
            neighbors,
            x,
            self.K_points,
            offsets,
            modulations,
            self.weight,
            self.point_influence,
            self.KP_influence,
            self.aggregation_mode,
        )

        if self.loss_mode == "fitting":
            self.internal_losses[self.OFFSET_LOSS_KEY] = self.loss_decay * (
                fitting_loss(sq_distances, self.kernel_radius) + repulsion_loss(K_points_deformed, self.kernel_radius)
            )
        elif self.loss_mode == "permissive":
            self.internal_losses[self.OFFSET_LOSS_KEY] = self.loss_decay * permissive_loss(
                K_points_deformed, self.radius
            )
        else:
            raise NotImplementedError(
                "Loss mode %s not recognised. Only permissive and fitting are valid" % self.loss_mode
            )

        return new_feat

    def get_internal_losses(self):
        return self.internal_losses

    def __repr__(self):
        return "KPConvDeformableLayer(InF: %i, OutF: %i, kernel_pts: %i, radius: %.2f, KP_influence: %s)" % (
            self.num_inputs,
            self.num_outputs,
            self.n_kernel_points,
            self.kernel_radius,
            self.KP_influence,
        )
