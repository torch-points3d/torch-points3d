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


class RSConvMapper(nn.Module):
    """[This class handles the special mechanism between the msg
        and the features of RSConv]
    """

    def __init__(self, down_conv_nn, channel_raising_nn, use_xyz, bn=True, *args, **kwargs):
        super(RSConvMapper, self).__init__()

        self._down_conv_nn = down_conv_nn
        self._channel_raising_nn = channel_raising_nn
        self._use_xyz = use_xyz

        if len(self._down_conv_nn) == 2:  # First layer
            self._first_layer = True
            f_in, f_intermediate, f_out = self._down_conv_nn[0]
            self.features_nn = pt_utils.SharedMLP(self._down_conv_nn[1], bn=bn)

        else:
            self._first_layer = False
            f_in, f_intermediate, f_out = self._down_conv_nn

        self.mlp_msg = pt_utils.SharedMLP([f_in, f_intermediate, f_out], bn=bn)

        self.mlp_out = nn.Conv1d(f_out, channel_raising_nn[-1], kernel_size=(1, 1))

    def forward(self, features, msg):
        """
        features  -- [B, C, num_points, nsamples]
        msg  -- [B, 10, num_points, nsamples]

        The 10 features comes from [distance: 1,
                                    coord_origin:3,
                                    coord_target:3,
                                    delta_origin_target:3]
        """

        # Transform msg
        msg = self.mlp_msg(msg)

        # If first_layer, augment features_size
        if self._first_layer:
            features = self.features_nn(features)

        return self.mlp_out(features * msg)


class SharedRSConv(nn.Module):
    """
    Input shape: (B, C_in, npoint, nsample)
    Output shape: (B, C_out, npoint)
    """

    def __init__(self, mapper: RSConvMapper):
        super(SharedRSConv, self).__init__()

        self._mapper = mapper

    def forward(self, aggr_features, centroids):
        """
        aggr_features  -- [B, 3 + 3 + C, num_points, nsamples]
        centroids  -- [B, 3, num_points, 1]
        """
        # Extract information to create message
        abs_coord = aggr_features[:, :3]  # absolute coordinates
        delta_x = aggr_features[:, 3:6]  # normalized coordinates
        features = aggr_features[:, 6:]

        nsample = abs_coord.shape[-1]
        coord_xi = centroids.repeat(1, 1, 1, nsample)  # (B, 3, npoint, nsample) centroid points

        distance = torch.norm(delta_x, p=2, dim=1).unsqueeze(1)  # Calculate distance

        # Create message by contenating distance, origin / target coords, delta coords
        h_xi_xj = torch.cat((distance, coord_xi, abs_coord, delta_x), dim=1)

        return self._mapper(features, h_xi_xj)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self._mapper.__repr__())


class RSConvMSGDown(BaseDenseConvolutionDown):
    def __init__(
        self,
        npoint=None,
        radii=None,
        nsample=None,
        down_conv_nn=None,
        channel_raising_nn=None,
        bn=True,
        use_xyz=True,
        **kwargs
    ):
        assert len(radii) == len(nsample)
        if len(radii) != len(down_conv_nn):
            log.warn("The down_conv_nn has a different size as radii. Make sure of have sharedMLP")
        super(RSConvMSGDown, self).__init__(
            DenseFPSSampler(num_to_sample=npoint), DenseRadiusNeighbourFinder(radii, nsample), **kwargs
        )

        self.use_xyz = use_xyz
        self.npoint = npoint
        self.mlps = nn.ModuleList()

        # https://github.com/Yochengliu/Relation-Shape-CNN/blob/6464eb8bb4efc686adec9da437112ef888e55684/utils/pointnet2_modules.py#L106
        mapper = RSConvMapper(down_conv_nn, channel_raising_nn, use_xyz=self.use_xyz)

        for _ in range(len(radii)):
            self.mlps.append(SharedRSConv(mapper))

    def _prepare_features(self, x, pos, new_pos, idx):
        new_pos_trans = pos.transpose(1, 2).contiguous()
        grouped_pos_absolute = tp.grouping_operation(new_pos_trans, idx)  # (B, 3, npoint, nsample)
        centroids = new_pos.transpose(1, 2).unsqueeze(-1)
        grouped_pos_normalized = grouped_pos_absolute - centroids

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

        return new_features, centroids

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
        aggr_features, centroids = self._prepare_features(x, pos, new_pos, radius_idx)
        new_features = self.mlps[scale_idx](aggr_features, centroids)  # (B, mlp[-1], npoint, nsample)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
        return new_features
