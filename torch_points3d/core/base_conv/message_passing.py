from abc import abstractmethod
from typing import *
import torch
from torch.nn import (
    Linear as Lin,
    ReLU,
    LeakyReLU,
    BatchNorm1d as BN,
    Dropout,
)
from torch_geometric.nn import (
    knn_interpolate,
    fps,
    radius,
    global_max_pool,
    global_mean_pool,
    knn,
)
from torch_geometric.data import Batch

from torch_points3d.core.base_conv.base_conv import *
from torch_points3d.core.common_modules import *
from torch_points3d.core.spatial_ops import *


def copy_from_to(data, batch):
    for key in data.keys:
        if key not in batch.keys:
            setattr(batch, key, getattr(data, key, None))


#################### THOSE MODULES IMPLEMENTS THE BASE MESSAGE_PASSING CONV API ############################


class BaseConvolutionDown(BaseConvolution):
    def __init__(self, sampler, neighbour_finder, *args, **kwargs):
        super(BaseConvolutionDown, self).__init__(sampler, neighbour_finder, *args, **kwargs)

        self._index = kwargs.get("index", None)

    def conv(self, x, pos, edge_index, batch):
        raise NotImplementedError

    def forward(self, data, **kwargs):
        batch_obj = Batch()
        x, pos, batch = data.x, data.pos, data.batch
        idx = self.sampler(pos, batch)
        row, col = self.neighbour_finder(pos, pos[idx], batch_x=batch, batch_y=batch[idx])
        edge_index = torch.stack([col, row], dim=0)
        batch_obj.idx = idx
        batch_obj.edge_index = edge_index

        batch_obj.x = self.conv(x, (pos[idx], pos), edge_index, batch)

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

        self._index = kwargs.get("index", None)

    def conv(self, x, pos, edge_index, batch):
        raise NotImplementedError

    def forward(self, data, **kwargs):
        batch_obj = Batch()
        x, pos, batch = data.x, data.pos, data.batch
        idx = self.sampler(pos, batch)
        batch_obj.idx = idx

        ms_x = []
        for scale_idx in range(self.neighbour_finder.num_scales):
            row, col = self.neighbour_finder(pos, pos[idx], batch_x=batch, batch_y=batch[idx], scale_idx=scale_idx,)
            edge_index = torch.stack([col, row], dim=0)

            ms_x.append(self.conv(x, (pos, pos[idx]), edge_index, batch))

        batch_obj.x = torch.cat(ms_x, -1)
        batch_obj.pos = pos[idx]
        batch_obj.batch = batch[idx]
        copy_from_to(data, batch_obj)
        return batch_obj


class BaseConvolutionUp(BaseConvolution):
    def __init__(self, neighbour_finder, *args, **kwargs):
        super(BaseConvolutionUp, self).__init__(None, neighbour_finder, *args, **kwargs)

        self._index = kwargs.get("index", None)
        self._skip = kwargs.get("skip", True)

    def conv(self, x, pos, pos_skip, batch, batch_skip, edge_index):
        raise NotImplementedError

    def forward(self, data, **kwargs):
        batch_obj = Batch()
        data, data_skip = data
        x, pos, batch = data.x, data.pos, data.batch
        x_skip, pos_skip, batch_skip = data_skip.x, data_skip.pos, data_skip.batch

        if self.neighbour_finder is not None:
            row, col = self.neighbour_finder(pos, pos_skip, batch, batch_skip)
            edge_index = torch.stack([col, row], dim=0)
        else:
            edge_index = None

        x = self.conv(x, pos, pos_skip, batch, batch_skip, edge_index)

        if x_skip is not None and self._skip:
            x = torch.cat([x, x_skip], dim=1)

        if hasattr(self, "nn"):
            batch_obj.x = self.nn(x)
        else:
            batch_obj.x = x
        copy_from_to(data_skip, batch_obj)
        return batch_obj


class GlobalBaseModule(torch.nn.Module):
    def __init__(self, nn, aggr="max", *args, **kwargs):
        super(GlobalBaseModule, self).__init__()
        self.nn = MLP(nn)
        self.pool = global_max_pool if aggr == "max" else global_mean_pool

    def forward(self, data, **kwargs):
        batch_obj = Batch()
        x, pos, batch = data.x, data.pos, data.batch
        if pos is not None:
            x = self.nn(torch.cat([x, pos], dim=1))
        x = self.pool(x, batch)
        batch_obj.x = x
        if pos is not None:
            batch_obj.pos = pos.new_zeros((x.size(0), 3))
        batch_obj.batch = torch.arange(x.size(0), device=batch.device)
        copy_from_to(data, batch_obj)
        return batch_obj


#################### COMMON MODULE ########################


class FPModule(BaseConvolutionUp):
    """ Upsampling module from PointNet++

    Arguments:
        k [int] -- number of nearest neighboors used for the interpolation
        up_conv_nn [List[int]] -- list of feature sizes for the uplconv mlp
    """

    def __init__(self, up_k, up_conv_nn, *args, **kwargs):
        super(FPModule, self).__init__(None)

        self.k = up_k
        bn_momentum = kwargs.get("bn_momentum", 0.1)
        self.nn = MLP(up_conv_nn, bn_momentum=bn_momentum, bias=False)

    def conv(self, x, pos, pos_skip, batch, batch_skip, *args):
        return knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)

    def extra_repr(self):
        return "Nb parameters: %i" % self.nb_params


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


class BaseResnetBlock(torch.nn.Module):
    def __init__(self, indim, outdim, convdim):
        """
            indim: size of x at the input
            outdim: desired size of x at the output
            convdim: size of x following convolution
        """
        torch.nn.Module.__init__(self)

        self.indim = indim
        self.outdim = outdim
        self.convdim = convdim

        self.features_downsample_nn = MLP([self.indim, self.outdim // 4])
        self.features_upsample_nn = MLP([self.convdim, self.outdim])

        self.shortcut_feature_resize_nn = MLP([self.indim, self.outdim])

        self.activation = ReLU()

    @property
    @abstractmethod
    def convs(self):
        pass

    def forward(self, data, **kwargs):
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
