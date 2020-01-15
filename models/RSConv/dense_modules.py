import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, BatchNorm2d
import torch_points as tp
import etw_pytorch_utils as pt_utils
from typing import Tuple, List

from models.dense_modules import *
from models.core_sampling_and_search import DenseFPSSampler, DenseRadiusNeighbourFinder

log = logging.getLogger(__name__)


class Mapper(nn.Module):
    def __init__(self, down_conv_nn, use_xyz, bn=True, *args, **kwargs):
        super(Mapper, self).__init__()

        self._down_conv_nn = down_conv_nn
        self._use_xyz = use_xyz

        f_in, f_intermediate, f_out = self._down_conv_nn

        self.squeeze_mlp = pt_utils.SharedMLP([f_in, f_intermediate], bn=bn)
        self.unsqueeze_mlp = pt_utils.SharedMLP([f_intermediate, f_out], bn=bn)

    def forward(self, x):
        return x


class SharedRSConv(nn.Module):
    """
    Input shape: (B, C_in, npoint, nsample)
    Output shape: (B, C_out, npoint)
    """

    def __init__(self, mapper):
        super(SharedRSConv, self).__init__()

        self._mapper = mapper

    def forward(self, input):

        new_features, centroid = input
        import pdb

        pdb.set_trace()

        """

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
        """

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self._mapper.__repr__())


class PointNetMSGDown(BaseDenseConvolutionDown):
    def __init__(self, npoint=None, radii=None, nsample=None, down_conv_nn=None, bn=True, use_xyz=True, **kwargs):
        assert len(radii) == len(nsample)
        if len(radii) != len(down_conv_nn):
            log.warn("The down_conv_nn has a different size as radii. Make sure of have sharedMLP")
        super(PointNetMSGDown, self).__init__(
            DenseFPSSampler(num_to_sample=npoint), DenseRadiusNeighbourFinder(radii, nsample), **kwargs
        )

        self.use_xyz = use_xyz
        self.npoint = npoint
        self.mlps = nn.ModuleList()

        # https://github.com/Yochengliu/Relation-Shape-CNN/blob/6464eb8bb4efc686adec9da437112ef888e55684/utils/pointnet2_modules.py#L106
        mapper = Mapper(down_conv_nn, use_xyz=self.use_xyz)

        for _ in range(len(radii)):
            self.mlps.append(SharedRSConv(mapper))

    def _prepare_features(self, x, pos, new_pos, idx):
        new_pos_trans = pos.transpose(1, 2).contiguous()
        grouped_pos_absolute = tp.grouping_operation(new_pos_trans, idx)  # (B, 3, npoint, nsample)
        grouped_pos_normalized = grouped_pos_absolute - new_pos.transpose(1, 2).unsqueeze(-1)

        if x is not None:
            grouped_features = tp.grouping_operation(x, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_pos_absolute, grouped_pos_normalized, grouped_features], dim=1
                )  # (B, 3 + 3 + C, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_pos_absolute

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
        new_features = self.mlps[scale_idx]([new_features, new_pos])  # (B, mlp[-1], npoint, nsample)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
        return new_features
