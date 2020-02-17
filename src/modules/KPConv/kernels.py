from enum import Enum
import torch
from torch.nn.parameter import Parameter

from .kernel_utils import kernel_point_optimization_debug, load_kernels
from .convolution_ops import *
from src.models.base_model import BaseInternalLossModule


class LossKPConvEnum(Enum):
    PERMISSIVE = 0
    FITTING = 1
    REPULSION = 2


def fitting_loss(sq_distance, radius):
    kpmin = sq_distance.min(dim=1)[0]
    normalised_kpmin = kpmin / (radius ** 2)
    return torch.mean(normalised_kpmin)


def repulsion_loss(deformed_kpoints, radius):
    deformed_kpoints / float(radius)
    n_points = deformed_kpoints.shape[1]
    repulsive_loss = 0
    for i in range(n_points):
        with torch.no_grad():
            other_points = torch.cat([deformed_kpoints[:, :i, :], deformed_kpoints[:, i + 1 :, :]], dim=1)
        distances = torch.sqrt(torch.sum((other_points - deformed_kpoints[:, i : i + 1, :]) ** 2, dim=-1))
        repulsion_force = torch.sum(torch.pow(torch.relu(1.5 - distances), 2), dim=1)
        repulsive_loss += torch.mean(repulsion_force)
    return repulsive_loss


def permissive_loss(deformed_kpoints, radius):
    """This loss is responsible to penalize deformed_kpoints to
    move outside from the radius defined for the convolution
    """
    norm_deformed_normalized = torch.norm(deformed_kpoints, p=2, dim=-1) / float(radius)
    permissive_loss = torch.mean(norm_deformed_normalized[norm_deformed_normalized > 1.0])
    return permissive_loss


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

    PERMISSIVE_LOSS_KEY = "permissive_loss"
    FITTING_LOSS_KEY = "fitting_loss"
    REPULSION_LOSS_KEY = "repulsion_loss"

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
        self.internal_losses = {self.PERMISSIVE_LOSS_KEY: 0.0, self.FITTING_LOSS_KEY: 0.0, self.REPULSION_LOSS_KEY: 0.0}
        self.loss_mode = loss_mode

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
            self.internal_losses[self.FITTING_LOSS_KEY] = fitting_loss(sq_distances, self.kernel_radius)
            self.internal_losses[self.REPULSION_LOSS_KEY] = repulsion_loss(K_points_deformed, self.point_influence)
        elif self.loss_mode == "permissive":
            self.internal_losses[self.PERMISSIVE_LOSS_KEY] = permissive_loss(K_points_deformed, self.kernel_radius)
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
