from abc import ABC, abstractmethod
from typing import *
import math
from functools import partial
from typing import Dict, Any
import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LeakyReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import knn_interpolate, fps, radius, global_max_pool, global_mean_pool, knn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter_
import torch_points as tp
import etw_pytorch_utils as pt_utils

import models.utils as utils
from models.core_sampling_and_search import BaseMSNeighbourFinder


def copy_from_to(data, batch):
    for key in data.keys:
        if key not in batch.keys:
            setattr(batch, key, getattr(data, key, None))


def MLP(channels, activation=ReLU()):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), activation, BN(channels[i]))
        for i in range(1, len(channels))
    ])


class BaseConvolution(ABC, torch.nn.Module):

    def __init__(self, sampler, neighbour_finder, *args, **kwargs):
        torch.nn.Module.__init__(self)

        self.sampler = sampler
        self.neighbour_finder = neighbour_finder


class BaseConvolutionDown(BaseConvolution):
    def __init__(self, sampler, neighbour_finder, *args, **kwargs):
        super(BaseConvolutionDown, self).__init__(sampler, neighbour_finder, *args, **kwargs)

        self._precompute_multi_scale = kwargs.get("precompute_multi_scale", None)
        self._index = kwargs.get("index", None)

    def conv(self, x, pos, edge_index, batch):
        raise NotImplementedError

    def forward(self, data):
        batch_obj = Batch()
        x, pos, batch = data.x, data.pos, data.batch
        if self._precompute_multi_scale:
            idx = getattr(data, "index_{}".format(self._index), None)
            edge_index = getattr(data, "edge_index_{}".format(self._index), None)
        else:
            idx = self.sampler(pos, batch)
            row, col = self.neighbour_finder(pos, pos[idx],  batch_x=batch, batch_y=batch[idx])
            edge_index = torch.stack([col, row], dim=0)
            batch_obj.idx = idx
            batch_obj.edge_index = edge_index

        batch_obj.x = self.conv(x, (pos, pos[idx]), edge_index, batch)

        batch_obj.pos = pos[idx]
        batch_obj.batch = batch[idx]
        copy_from_to(data, batch_obj)
        return batch_obj


class BaseMSConvolutionDown(BaseConvolution):
    """ Multiscale convolution down (also supports single scale). Convolution kernel is shared accross the scales

        Arguments:
            sampler  -- Strategy for sampling the input clouds
            neighbour_finder -- Multiscale strategy for finding neighbours
    """

    def __init__(self, sampler, neighbour_finder: BaseMSNeighbourFinder, *args, **kwargs):
        super(BaseMSConvolutionDown, self).__init__(sampler, neighbour_finder, *args, **kwargs)

        self._precompute_multi_scale = kwargs.get("precompute_multi_scale", None)
        self._index = kwargs.get("index", None)

    def conv(self, x, pos, edge_index, batch):
        raise NotImplementedError

    def forward(self, data):
        batch_obj = Batch()
        x, pos, batch = data.x, data.pos, data.batch
        if self._precompute_multi_scale:
            idx = getattr(data, "idx_{}".format(self._index), None)
        else:
            idx = self.sampler(pos, batch)
            batch_obj.idx = idx

        ms_x = []
        for scale_idx in range(self.neighbour_finder.num_scales):
            if self._precompute_multi_scale:
                edge_index = getattr(data, "edge_index_{}_{}".format(self._index, scale_idx), None)
            else:
                row, col = self.neighbour_finder(pos, pos[idx], batch_x=batch, batch_y=batch[idx], scale_idx=scale_idx)
                edge_index = torch.stack([col, row], dim=0)

            ms_x.append(self.conv(x, (pos, pos[idx]), edge_index, batch))

        batch_obj.x = torch.cat(ms_x, -1)
        batch_obj.pos = pos[idx]
        batch_obj.batch = batch[idx]
        copy_from_to(data, batch_obj)
        return batch_obj


class BaseDenseConvolutionDown(BaseConvolution):
    """ Multiscale convolution down (also supports single scale). Convolution kernel is shared accross the scales

        Arguments:
            sampler  -- Strategy for sampling the input clouds
            neighbour_finder -- Multiscale strategy for finding neighbours
    """

    def __init__(self, sampler, neighbour_finder: BaseMSNeighbourFinder, *args, **kwargs):
        super(BaseDenseConvolutionDown, self).__init__(sampler, neighbour_finder, *args, **kwargs)

        self._precompute_multi_scale = kwargs.get("precompute_multi_scale", None)
        self._index = kwargs.get("index", None)

    def conv(self, x, pos, new_pos, radius_idx):
        """ Implements a Dense convolution where radius_idx represents
        the indexes of the points in x and pos to be agragated into the new feature
        for each point in new_pos

        Arguments:
            x -- Previous features [B, N, C]
            pos -- Previous positions [B, N, 3]
            new_pos  -- Sampled positions [B, npoints, 3]
            radius_idx -- Indexes to group [B, npoints, nsample]
        """
        raise NotImplementedError

    def forward(self, data):
        batch_obj = Batch()
        x, pos = data.x, data.pos
        if self._precompute_multi_scale:
            idx = getattr(data, "idx_{}".format(self._index), None)
        else:
            idx = self.sampler(pos)
            batch_obj.idx = idx

        pos_flipped = pos.transpose(1, 2).contiguous()
        new_pos = tp.gather_operation(pos_flipped, idx).transpose(1, 2).contiguous()

        ms_x = []
        for scale_idx in range(self.neighbour_finder.num_scales):
            if self._precompute_multi_scale:
                raise NotImplementedError()
            else:
                radius_idx = self.neighbour_finder(pos, new_pos, scale_idx=scale_idx)

            ms_x.append(self.conv(x, pos, new_pos, radius_idx))

        batch_obj.x = torch.cat(ms_x, -1)
        batch_obj.pos = new_pos
        copy_from_to(data, batch_obj)
        return batch_obj


class BaseConvolutionUp(BaseConvolution):
    def __init__(self, neighbour_finder, *args, **kwargs):
        super(BaseConvolutionUp, self).__init__(None, neighbour_finder, *args, **kwargs)

        self._precompute_multi_scale = kwargs.get("precompute_multi_scale", None)
        self._index = kwargs.get("index", None)
        self._skip = kwargs.get("skip", True)

    def conv(self, x, pos, pos_skip, batch, batch_skip, edge_index):
        raise NotImplementedError

    def forward(self, data):
        batch_obj = Batch()
        data, data_skip = data
        x, pos, batch = data.x, data.pos, data.batch
        x_skip, pos_skip, batch_skip = data_skip.x, data_skip.pos, data_skip.batch

        if self.neighbour_finder is not None:
            if self._precompute_multi_scale:  # TODO For now, it uses the one calculated during down steps
                edge_index = getattr(data_skip, "edge_index_{}".format(self._index), None)
                col, row = edge_index
                edge_index = torch.stack([row, col], dim=0)
            else:
                row, col = self.neighbour_finder(pos, pos_skip, batch, batch_skip)
                edge_index = torch.stack([col, row], dim=0)
        else:
            edge_index = None

        x = self.conv(x, pos, pos_skip, batch, batch_skip, edge_index)

        if x_skip is not None and self._skip:
            x = torch.cat([x, x_skip], dim=1)

        if hasattr(self, 'nn'):
            batch_obj.x = self.nn(x)
        else:
            batch_obj.x = x
        copy_from_to(data_skip, batch_obj)
        return batch_obj


class BaseDenseConvolutionUp(BaseConvolution):
    def __init__(self, neighbour_finder, *args, **kwargs):
        super(BaseDenseConvolutionUp, self).__init__(None, neighbour_finder, *args, **kwargs)

        self._precompute_multi_scale = kwargs.get("precompute_multi_scale", None)
        self._index = kwargs.get("index", None)
        self._skip = kwargs.get("skip", True)

    def conv(self, x, x_skip, pos, pos_skip, batch, batch_skip):
        raise NotImplementedError

    def forward(self, data):
        batch_obj = Batch()
        data, data_skip = data
        x, pos, batch = data.x, data.pos, data.batch
        x_skip, pos_skip, batch_skip = data_skip.x, data_skip.pos, data_skip.batch

        x = self.conv(x, x_skip, pos, pos_skip, batch, batch_skip)

        if x_skip is not None and self._skip:
            if x.shape[-1] == x_skip.shape[-1]:
                x = torch.cat([x, x_skip], dim=1)
            else:
                x_skip = x_skip.transpose(1, 2).contiguous()
                x = torch.cat([x, x_skip], dim=1)

        x = x.unsqueeze(-1)

        if hasattr(self, 'nn'):
            batch_obj.x = self.nn(x)
        else:
            batch_obj.x = x
        copy_from_to(data_skip, batch_obj)
        return batch_obj


class GlobalBaseModule(torch.nn.Module):
    def __init__(self, nn, aggr='max'):
        super(GlobalBaseModule, self).__init__()
        self.nn = MLP(nn)
        self.pool = global_max_pool if aggr == "max" else global_mean_pool

    def forward(self, data):
        batch_obj = Batch()
        x, pos, batch = data.x, data.pos, data.batch
        x = self.nn(torch.cat([x, pos], dim=1))
        x = self.pool(x, batch)
        batch_obj.x = x
        batch_obj.pos = pos.new_zeros((x.size(0), 3))
        batch_obj.batch = torch.arange(x.size(0), device=batch.device)
        copy_from_to(data, batch_obj)
        return batch_obj


class GlobalDenseBaseModule(torch.nn.Module):
    def __init__(self, nn, aggr='max'):
        super(GlobalDenseBaseModule, self).__init__()
        self.nn = pt_utils.SharedMLP(nn)

    def forward(self, data):
        batch_obj = Batch()
        x, pos, batch = data.x, data.pos, data.batch
        pos_flipped = pos.transpose(1, 2).contiguous()
        x = self.nn(torch.cat([x, pos_flipped], dim=1).unsqueeze(-1))
        x = x.squeeze().max(-1)[0]
        batch_obj.x = x
        batch_obj.pos = pos.new_zeros((x.size(0), 3, 1))
        batch_obj.batch = torch.arange(x.size(0), device=x.device)
        copy_from_to(data, batch_obj)
        return batch_obj

#################### COMMON MODULE ########################


class FPModule(BaseConvolutionUp):
    """ Upsampling module from PointNet++

    Arguments:
        k [int] -- number of nearest neighboors used for the interpolation
        up_conv_nn [List[int]] -- list of feature sizes for the uplconv mlp

    Returns:
        [type] -- [description]
    """

    def __init__(self, up_k, up_conv_nn, nb_feature=None, **kwargs):
        super(FPModule, self).__init__(None)

        self.k = up_k
        self.nn = MLP(up_conv_nn)

    def conv(self, x, pos, pos_skip, batch, batch_skip, *args):
        return knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)


class DenseFPModule(BaseDenseConvolutionUp):
    def __init__(self, up_k, up_conv_nn, nb_feature=None, **kwargs):
        super(DenseFPModule, self).__init__(None)

        self.k = up_k
        self.nn = pt_utils.SharedMLP(up_conv_nn)

    def conv(self, x, x_skip, pos, pos_skip, batch, batch_skip):
        # unknown, known, unknow_feats, known_feats
        # torch.Size([32, 64, 3]) torch.Size([32, 16, 3]) torch.Size([32, 512, 64]) torch.Size([32, 1024, 16])

        dist, idx = tp.three_nn(pos_skip, pos)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm

        x = x.squeeze(-1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(-1)

        #print(x.shape, x_skip.shape, pos.shape, pos_skip.shape)
        interpolated_feats = tp.three_interpolate(x, idx, weight)
        return interpolated_feats

########################## BASE RESIDUAL DOWN #####################


class BaseResnetBlockDown(BaseConvolutionDown):

    def __init__(self, sampler, neighbour_finder, *args, **kwargs):
        super(BaseResnetBlockDown, self).__init__(sampler, neighbour_finder, *args, **kwargs)

        in_features, out_features, conv_features = kwargs.get("down_conv_nn", None)

        self.in_features = in_features
        self.out_features = out_features
        self.conv_features = conv_features

        self.features_downsample_nn = MLP([self.in_features, self.conv_features])

        self.features_upsample_nn = MLP([self.conv_features, self.out_features])
        self.shortcut_feature_resize_nn = MLP([self.in_features, self.out_features])

    def convs(self, x, pos, edge_index):
        raise NotImplementedError

    def conv(self, x, pos, edge_index):
        shortcut = x
        x = self.features_downsample_nn(x)
        x, pos, edge_index, idx = self.convs(x, pos, edge_index)
        x = self.features_upsample_nn(x)
        if idx is not None:
            shortcut = shortcut[idx]
        shortcut = self.shortcut_feature_resize_nn(shortcut)
        x = shortcut + x
        return x


class BaseResnetBlock(ABC, torch.nn.Module):

    def __init__(self, indim, outdim, convdim):
        '''
            indim: size of x at the input
            outdim: desired size of x at the output
            convdim: size of x following convolution
        '''
        torch.nn.Module.__init__(self)

        self.indim = indim
        self.outdim = outdim
        self.convdim = convdim

        self.features_downsample_nn = MLP([self.indim, self.outdim//4])
        self.features_upsample_nn = MLP([self.convdim, self.outdim])

        self.shortcut_feature_resize_nn = MLP([self.indim, self.outdim])

        self.activation = ReLU()

    @property
    @abstractmethod
    def convs(self):
        pass

    def forward(self, data):
        batch_obj = Batch()
        x = data.x  # (N, indim)
        shortcut = x  # (N, indim)
        x = self.features_downsample_nn(x)  # (N, outdim//4)
        # if this is an identity resnet block, idx will be None
        data = self.convs(data)  # (N', convdim)
        x = data.x
        idx = data.idx
        x = self.features_upsample_nn(x)  # (N', outdim)
        if idx is not None:
            shortcut = shortcut[idx]  # (N', indim)
        shortcut = self.shortcut_feature_resize_nn(shortcut)  # (N', outdim)
        x = shortcut + x
        batch_obj.x = x
        batch_obj.pos = data.pos
        batch_obj.batch = data.batch
        copy_from_to(data, batch_obj)
        return batch_obj
