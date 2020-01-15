import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, BatchNorm2d
import torch_points as tp
import etw_pytorch_utils as pt_utils
from typing import Tuple, List

from models.dense_modules import *
from models.core_sampling_and_search import DenseFPSSampler, DenseRadiusNeighbourFinder


class RSConv(nn.Module):
    """
    Input shape: (B, C_in, npoint, nsample)
    Output shape: (B, C_out, npoint)
    """

    def __init__(
        self, C_in, C_out, activation=nn.ReLU(inplace=True), mapping=None, relation_prior=1, first_layer=False
    ):
        super(RSConv, self).__init__()
        self.bn_rsconv = nn.BatchNorm2d(C_in) if not first_layer else nn.BatchNorm2d(16)
        self.bn_channel_raising = nn.BatchNorm1d(C_out)
        self.bn_xyz_raising = nn.BatchNorm2d(16)
        if first_layer:
            self.bn_mapping = nn.BatchNorm2d(math.floor(C_out / 2))
        else:
            self.bn_mapping = nn.BatchNorm2d(math.floor(C_out / 4))
        self.activation = activation
        self.relation_prior = relation_prior
        self.first_layer = first_layer
        self.mapping_func1 = mapping[0]
        self.mapping_func2 = mapping[1]
        self.cr_mapping = mapping[2]
        if first_layer:
            self.xyz_raising = mapping[3]

    def forward(self, input):  # input: (B, 3 + 3 + C_in, npoint, centroid + nsample)

        x = input[:, 3:, :, :]  # (B, C_in, npoint, nsample+1), input features
        C_in = x.size()[1]
        nsample = x.size()[3]
        if self.relation_prior == 2:
            abs_coord = input[:, 0:2, :, :]
            delta_x = input[:, 3:5, :, :]
            zero_vec = Variable(torch.zeros(x.size()[0], 1, x.size()[2], nsample).cuda())
        else:
            abs_coord = input[:, 0:3, :, :]  # (B, 3, npoint, nsample+1), absolute coordinates
            delta_x = input[:, 3:6, :, :]  # (B, 3, npoint, nsample+1), normalized coordinates

        coord_xi = abs_coord[:, :, :, 0:1].repeat(1, 1, 1, nsample)  # (B, 3, npoint, nsample),  centroid point
        h_xi_xj = torch.norm(delta_x, p=2, dim=1).unsqueeze(1)
        if self.relation_prior == 1:
            h_xi_xj = torch.cat((h_xi_xj, coord_xi, abs_coord, delta_x), dim=1)
        elif self.relation_prior == 2:
            h_xi_xj = torch.cat((h_xi_xj, coord_xi, zero_vec, abs_coord, zero_vec, delta_x, zero_vec), dim=1)
        del coord_xi, abs_coord, delta_x

        h_xi_xj = self.mapping_func2(self.activation(self.bn_mapping(self.mapping_func1(h_xi_xj))))
        if self.first_layer:
            x = self.activation(self.bn_xyz_raising(self.xyz_raising(x)))
        x = F.max_pool2d(self.activation(self.bn_rsconv(torch.mul(h_xi_xj, x))), kernel_size=(1, nsample)).squeeze(
            3
        )  # (B, C_in, npoint)
        del h_xi_xj
        x = self.activation(self.bn_channel_raising(self.cr_mapping(x)))

        return x


class RSConvLayer(nn.Sequential):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        activation=nn.ReLU(inplace=True),
        conv=RSConv,
        mapping=None,
        relation_prior=1,
        first_layer=False,
    ):
        super(RSConvLayer, self).__init__()

        conv_unit = conv(
            in_size,
            out_size,
            activation=activation,
            mapping=mapping,
            relation_prior=relation_prior,
            first_layer=first_layer,
        )

        self.add_module("RS_Conv", conv_unit)


class SharedRSConv(nn.Sequential):
    def __init__(
        self, args: List[int], *, activation=nn.ReLU(inplace=True), mapping=None, relation_prior=1, first_layer=False
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                "RSConvLayer{}".format(i),
                RSConvLayer(
                    args[i],
                    args[i + 1],
                    activation=activation,
                    mapping=mapping,
                    relation_prior=relation_prior,
                    first_layer=first_layer,
                ),
            )


class PointNetMSGDown(BaseDenseConvolutionDown):
    def __init__(self, npoint=None, radii=None, nsample=None, down_conv_nn=None, bn=True, use_xyz=True, **kwargs):
        assert len(radii) == len(nsample) == len(down_conv_nn)
        super(PointNetMSGDown, self).__init__(
            DenseFPSSampler(num_to_sample=npoint), DenseRadiusNeighbourFinder(radii, nsample), **kwargs
        )
        self.use_xyz = use_xyz
        self.npoint = npoint
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            mlp_spec = down_conv_nn[i]
            if self.use_xyz:
                mlp_spec[0] += 3
            self.mlps.append(pt_utils.SharedRSConv(down_conv_nn[i], bn=bn))

    def _prepare_features(self, x, pos, new_pos, idx):
        new_pos_trans = pos.transpose(1, 2).contiguous()
        grouped_pos = tp.grouping_operation(new_pos_trans, idx)  # (B, 3, npoint, nsample)
        grouped_pos -= new_pos.transpose(1, 2).unsqueeze(-1)

        if x is not None:
            grouped_features = tp.grouping_operation(x, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_pos, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_pos

        return new_features

    def conv(self, x, pos, new_pos, radius_idx, scale_idx):
        """ Implements a Dense convolution where radius_idx represents
        the indexes of the points in x and pos to be agragated into the new feature
        for each point in new_pos

        Arguments:
            x -- Previous features [B, N, C]
            pos -- Previous positions [B, N, 3]
            new_pos  -- Sampled positions [B, npoints, 3]
            radius_idx -- Indexes to group [B, npoints, nsample]
            scale_idx -- Scale index in multiscale convolutional layers
        Returns:
            new_x -- Features after passing trhough the MLP [B, mlp[-1], npoints]
        """
        assert scale_idx < len(self.mlps)
        new_features = self._prepare_features(x, pos, new_pos, radius_idx)
        new_features = self.mlps[scale_idx](new_features)  # (B, mlp[-1], npoint, nsample)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
        return new_features
