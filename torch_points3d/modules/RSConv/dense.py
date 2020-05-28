import logging

import torch
from torch.nn import Sequential
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
import torch_points_kernels as tp

from torch_points3d.core.base_conv.dense import *
from torch_points3d.core.common_modules.dense_modules import MLP2D
from torch_points3d.core.spatial_ops import DenseFPSSampler, DenseRadiusNeighbourFinder
from torch_points3d.utils.colors import COLORS

log = logging.getLogger(__name__)


class RSConvMapper(nn.Module):
    """[This class handles the special mechanism between the msg
        and the features of RSConv]
    """

    def __init__(self, down_conv_nn, use_xyz, bn=True, activation=nn.LeakyReLU(negative_slope=0.01), *args, **kwargs):
        super(RSConvMapper, self).__init__()

        self._down_conv_nn = down_conv_nn
        self._use_xyz = use_xyz

        self.nn = nn.ModuleDict()

        if len(self._down_conv_nn) == 2:  # First layer
            self._first_layer = True
            f_in, f_intermediate, f_out = self._down_conv_nn[0]
            self.nn["features_nn"] = MLP2D(self._down_conv_nn[1], bn=bn, bias=False)

        else:
            self._first_layer = False
            f_in, f_intermediate, f_out = self._down_conv_nn

        self.nn["mlp_msg"] = MLP2D([f_in, f_intermediate, f_out], bn=bn, bias=False)

        self.nn["norm"] = Sequential(*[nn.BatchNorm2d(f_out), activation])

        self._f_out = f_out

    @property
    def f_out(self):
        return self._f_out

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
        msg = self.nn["mlp_msg"](msg)

        # If first_layer, augment features_size
        if self._first_layer:
            features = self.nn["features_nn"](features)

        return self.nn["norm"](torch.mul(features, msg))


class SharedRSConv(nn.Module):
    """
    Input shape: (B, C_in, npoint, nsample)
    Output shape: (B, C_out, npoint)
    """

    def __init__(self, mapper: RSConvMapper, radius):
        super(SharedRSConv, self).__init__()

        self._mapper = mapper
        self._radius = radius

    def forward(self, aggr_features, centroids):
        """
        aggr_features  -- [B, 3 + 3 + C, num_points, nsamples]
        centroids  -- [B, 3, num_points, 1]
        """
        # Extract information to create message
        abs_coord = aggr_features[:, :3]  # absolute coordinates
        delta_x = aggr_features[:, 3:6]  # normalized coordinates
        features = aggr_features[:, 3:]

        nsample = abs_coord.shape[-1]
        coord_xi = centroids.repeat(1, 1, 1, nsample)  # (B, 3, npoint, nsample) centroid points

        distance = torch.norm(delta_x, p=2, dim=1).unsqueeze(1)  # Calculate distance

        # Create message by contenating distance, origin / target coords, delta coords
        h_xi_xj = torch.cat((distance, coord_xi, abs_coord, delta_x), dim=1)

        return self._mapper(features, h_xi_xj)

    def __repr__(self):
        return "{}(radius={})".format(self.__class__.__name__, self._radius)


class RSConvSharedMSGDown(BaseDenseConvolutionDown):
    def __init__(
        self,
        npoint=None,
        radii=None,
        nsample=None,
        down_conv_nn=None,
        channel_raising_nn=None,
        bn=True,
        use_xyz=True,
        activation=nn.ReLU(),
        **kwargs
    ):
        assert len(radii) == len(nsample)
        if len(radii) != len(down_conv_nn):
            log.warn("The down_conv_nn has a different size as radii. Make sure of have SharedRSConv")
        super(RSConvSharedMSGDown, self).__init__(
            DenseFPSSampler(num_to_sample=npoint), DenseRadiusNeighbourFinder(radii, nsample), **kwargs
        )

        self.use_xyz = use_xyz
        self.npoint = npoint
        self.mlps = nn.ModuleList()

        # https://github.com/Yochengliu/Relation-Shape-CNN/blob/6464eb8bb4efc686adec9da437112ef888e55684/utils/pointnet2_modules.py#L106
        self._mapper = RSConvMapper(down_conv_nn, activation=activation, use_xyz=self.use_xyz)

        self.mlp_out = Sequential(
            *[
                Conv1d(channel_raising_nn[0], channel_raising_nn[-1], kernel_size=1, stride=1, bias=True),
                nn.BatchNorm1d(channel_raising_nn[-1]),
                activation,
            ]
        )

        for i in range(len(radii)):
            self.mlps.append(SharedRSConv(self._mapper, radii[i]))

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
            new_features = torch.cat(
                [grouped_pos_absolute, grouped_pos_normalized], dim=1
            )  # (B, 3 + 3 npoint, nsample)

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
        new_features = self.mlp_out(new_features.squeeze(-1))  # (B, mlp[-1], npoint)
        return new_features

    def __repr__(self):
        return "{}({}, shared: {} {}, {} {})".format(
            self.__class__.__name__,
            self.mlps.__repr__(),
            COLORS.Cyan,
            self.mlp_out.__repr__(),
            self._mapper.__repr__(),
            COLORS.END_TOKEN,
        )


######################################################################


class OriginalRSConv(nn.Module):
    """
    Input shape: (B, C_in, npoint, nsample)
    Output shape: (B, C_out, npoint)
    """

    def __init__(self, mapping=None, first_layer=False, radius=None, activation=nn.ReLU(inplace=True)):
        super(OriginalRSConv, self).__init__()

        self.nn = nn.ModuleList()

        self._radius = radius

        self.mapping_func1 = mapping[0]
        self.mapping_func2 = mapping[1]
        self.cr_mapping = mapping[2]

        self.first_layer = first_layer

        if first_layer:
            self.xyz_raising = mapping[3]
            self.bn_xyz_raising = nn.BatchNorm2d(self.xyz_raising.out_channels)
            self.nn.append(self.bn_xyz_raising)

        self.bn_mapping = nn.BatchNorm2d(self.mapping_func1.out_channels)
        self.bn_rsconv = nn.BatchNorm2d(self.cr_mapping.in_channels)
        self.bn_channel_raising = nn.BatchNorm1d(self.cr_mapping.out_channels)

        self.nn.append(self.bn_mapping)
        self.nn.append(self.bn_rsconv)
        self.nn.append(self.bn_channel_raising)

        self.activation = activation

    def forward(self, input):  # input: (B, 3 + 3 + C_in, npoint, centroid + nsample)

        x = input[:, 3:, :, :]  # (B, C_in, npoint, nsample+1), input features
        nsample = x.size()[3]
        abs_coord = input[:, 0:3, :, :]  # (B, 3, npoint, nsample+1), absolute coordinates
        delta_x = input[:, 3:6, :, :]  # (B, 3, npoint, nsample+1), normalized coordinates

        coord_xi = abs_coord[:, :, :, 0:1].repeat(1, 1, 1, nsample)  # (B, 3, npoint, nsample),  centroid point
        h_xi_xj = torch.norm(delta_x, p=2, dim=1).unsqueeze(1)
        h_xi_xj = torch.cat((h_xi_xj, coord_xi, abs_coord, delta_x), dim=1)

        h_xi_xj = self.mapping_func2(self.activation(self.bn_mapping(self.mapping_func1(h_xi_xj))))

        if self.first_layer:
            x = self.activation(self.bn_xyz_raising(self.xyz_raising(x)))
        x = F.max_pool2d(self.activation(self.bn_rsconv(torch.mul(h_xi_xj, x))), kernel_size=(1, nsample)).squeeze(
            3
        )  # (B, C_in, npoint)
        x = self.activation(self.bn_channel_raising(self.cr_mapping(x)))
        return x

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.nn.__repr__())


class RSConvOriginalMSGDown(BaseDenseConvolutionDown):
    def __init__(
        self,
        npoint=None,
        radii=None,
        nsample=None,
        down_conv_nn=None,
        channel_raising_nn=None,
        bn=True,
        bias=True,
        use_xyz=True,
        activation=nn.ReLU(),
        **kwargs
    ):
        assert len(radii) == len(nsample)
        if len(radii) != len(down_conv_nn):
            log.warning("The down_conv_nn has a different size as radii. Make sure of have SharedRSConv")
        super(RSConvOriginalMSGDown, self).__init__(
            DenseFPSSampler(num_to_sample=npoint), DenseRadiusNeighbourFinder(radii, nsample), **kwargs
        )

        self.use_xyz = use_xyz
        self.mlps = nn.ModuleList()
        self.mappings = nn.ModuleList()

        self._first_layer = True if len(down_conv_nn) == 2 else False

        if self._first_layer:
            C_in, C_intermediate, C_out = down_conv_nn[0]
            feat_in, f_out = down_conv_nn[-1]
            xyz_raising = nn.Conv2d(
                in_channels=feat_in, out_channels=f_out, kernel_size=(1, 1), stride=(1, 1), bias=bias,
            )
            nn.init.kaiming_normal_(xyz_raising.weight)
            if bias:
                nn.init.constant_(xyz_raising.bias, 0)
        else:
            C_in, C_intermediate, C_out = down_conv_nn

        mapping_func1 = nn.Conv2d(
            in_channels=C_in, out_channels=C_intermediate, kernel_size=(1, 1), stride=(1, 1), bias=bias,
        )
        mapping_func2 = nn.Conv2d(
            in_channels=C_intermediate, out_channels=C_out, kernel_size=(1, 1), stride=(1, 1), bias=bias,
        )

        nn.init.kaiming_normal_(mapping_func1.weight)
        nn.init.kaiming_normal_(mapping_func2.weight)
        if bias:
            nn.init.constant_(mapping_func1.bias, 0)
            nn.init.constant_(mapping_func2.bias, 0)

        # channel raising mapping
        cr_mapping = nn.Conv1d(
            in_channels=channel_raising_nn[0], out_channels=channel_raising_nn[1], kernel_size=1, stride=1, bias=bias,
        )
        nn.init.kaiming_normal_(cr_mapping.weight)
        nn.init.constant_(cr_mapping.bias, 0)

        if self._first_layer:
            mapping = [mapping_func1, mapping_func2, cr_mapping, xyz_raising]
        elif npoint is not None:
            mapping = [mapping_func1, mapping_func2, cr_mapping]

        for m in mapping:
            self.mappings.append(m)

        for radius in radii:
            self.mlps.append(OriginalRSConv(mapping=mapping, first_layer=self._first_layer, radius=radius))

    def _prepare_features(
        self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None, idx: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = tp.grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        raw_grouped_xyz = grouped_xyz
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = tp.grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [raw_grouped_xyz, grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + 3 + C, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = torch.cat([raw_grouped_xyz, grouped_xyz], dim=1)

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
        aggr_features = self._prepare_features(pos, new_pos, x, radius_idx)
        new_features = self.mlps[scale_idx](
            aggr_features
        )  # (B, 3 + 3 + C, npoint, nsample) -> (B, mlp[-1], npoint, nsample)
        return new_features

    def __repr__(self):
        return "{}: {} ({}, shared: {} {} {})".format(
            self.__class__.__name__,
            self.nb_params,
            self.mlps.__repr__(),
            COLORS.Cyan,
            self.mappings.__repr__(),
            COLORS.END_TOKEN,
        )


class RSConvMSGDown(BaseDenseConvolutionDown):
    def __init__(
        self,
        npoint=None,
        radii=None,
        nsample=None,
        down_conv_nn=None,
        channel_raising_nn=None,
        bn=True,
        bias=True,
        use_xyz=True,
        activation=nn.ReLU(),
        **kwargs
    ):
        assert len(radii) == len(nsample)
        if len(radii) != len(down_conv_nn):
            log.warning("The down_conv_nn has a different size as radii. Make sure to have sharedMLP")
        super(RSConvMSGDown, self).__init__(
            DenseFPSSampler(num_to_sample=npoint), DenseRadiusNeighbourFinder(radii, nsample), **kwargs
        )

        self.use_xyz = use_xyz
        self.npoint = npoint
        self.mlps = nn.ModuleList()

        # https://github.com/Yochengliu/Relation-Shape-CNN/blob/6464eb8bb4efc686adec9da437112ef888e55684/utils/pointnet2_modules.py#L106

        self.mlp_out = Sequential(
            *[
                Conv1d(channel_raising_nn[0], channel_raising_nn[-1], kernel_size=1, stride=1, bias=True),
                nn.BatchNorm1d(channel_raising_nn[-1]),
                activation,
            ]
        )

        for i in range(len(radii)):
            mapper = RSConvMapper(down_conv_nn, activation=activation, use_xyz=self.use_xyz)
            self.mlps.append(SharedRSConv(mapper, radii[i]))

        self._mapper = mapper

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
            new_features = torch.cat(
                [grouped_pos_absolute, grouped_pos_normalized], dim=1
            )  # (B, 3 + 3 npoint, nsample)

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
        new_features = self.mlp_out(new_features.squeeze(-1))  # (B, mlp[-1], npoint)
        return new_features

    def __repr__(self):
        return "{}({}, shared: {} {}, {} {})".format(
            self.__class__.__name__,
            self.mlps.__repr__(),
            COLORS.Cyan,
            self.mlp_out.__repr__(),
            self._mapper.__repr__(),
            COLORS.END_TOKEN,
        )
