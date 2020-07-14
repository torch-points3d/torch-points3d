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

from torch_points3d.core.spatial_ops import *
from .base_conv import BaseConvolution
from torch_points3d.core.common_modules.base_modules import BaseModule
from torch_points3d.core.common_modules import MLP


#################### THOSE MODULES IMPLEMENTS THE BASE PARTIAL_DENSE CONV API ############################


def copy_from_to(data, batch):
    for key in data.keys:
        if key not in batch.keys:
            setattr(batch, key, getattr(data, key, None))


class BasePartialDenseConvolutionDown(BaseConvolution):

    CONV_TYPE = ConvolutionFormat.PARTIAL_DENSE.value

    def __init__(self, sampler, neighbour_finder, *args, **kwargs):
        super(BasePartialDenseConvolutionDown, self).__init__(sampler, neighbour_finder, *args, **kwargs)

        self._index = kwargs.get("index", None)

    def conv(self, x, pos, x_neighbour, pos_centered_neighbour, idx_neighbour, idx_sampler):
        """ Generic down convolution for partial dense data

        Arguments:
            x [N, C] -- features
            pos [N, 3] -- positions
            x_neighbour [N, n_neighbours, C] -- features of the neighbours of each point in x
            pos_centered_neighbour [N, n_neighbours, 3]  -- position of the neighbours of x_i centred on x_i
            idx_neighbour [N, n_neighbours] -- indices of the neighbours of each point in pos
            idx_sampler [n]  -- points to keep for the output

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def forward(self, data, **kwargs):
        batch_obj = Batch()
        x, pos, batch = data.x, data.pos, data.batch
        idx_sampler = self.sampler(pos=pos, x=x, batch=batch)

        idx_neighbour = self.neighbour_finder(pos, pos, batch_x=batch, batch_y=batch)

        shadow_x = torch.full((1,) + x.shape[1:], self.shadow_features_fill).to(x.device)
        shadow_points = torch.full((1,) + pos.shape[1:], self.shadow_points_fill_).to(x.device)

        x = torch.cat([x, shadow_x], dim=0)
        pos = torch.cat([pos, shadow_points], dim=0)

        x_neighbour = x[idx_neighbour]
        pos_centered_neighbour = pos[idx_neighbour] - pos[:-1].unsqueeze(1)  # Centered the points, no shadow point

        batch_obj.x = self.conv(x, pos, x_neighbour, pos_centered_neighbour, idx_neighbour, idx_sampler)

        batch_obj.pos = pos[idx_sampler]
        batch_obj.batch = batch[idx_sampler]
        copy_from_to(data, batch_obj)
        return batch_obj


class GlobalPartialDenseBaseModule(torch.nn.Module):
    def __init__(self, nn, aggr="max", *args, **kwargs):
        super(GlobalPartialDenseBaseModule, self).__init__()

        self.nn = MLP(nn)
        self.pool = global_max_pool if aggr == "max" else global_mean_pool

    def forward(self, data, **kwargs):
        batch_obj = Batch()
        x, pos, batch = data.x, data.pos, data.batch
        x = self.nn(torch.cat([x, pos], dim=1))
        x = self.pool(x, batch)
        batch_obj.x = x
        batch_obj.pos = pos.new_zeros((x.size(0), 3))
        batch_obj.batch = torch.arange(x.size(0), device=x.device)
        copy_from_to(data, batch_obj)
        return batch_obj


class FPModule_PD(BaseModule):
    """ Upsampling module from PointNet++
    Arguments:
        k [int] -- number of nearest neighboors used for the interpolation
        up_conv_nn [List[int]] -- list of feature sizes for the uplconv mlp
    """

    def __init__(self, up_k, up_conv_nn, *args, **kwargs):
        super(FPModule_PD, self).__init__()
        self.upsample_op = KNNInterpolate(up_k)
        bn_momentum = kwargs.get("bn_momentum", 0.1)
        self.nn = MLP(up_conv_nn, bn_momentum=bn_momentum, bias=False)

    def forward(self, data, precomputed=None, **kwargs):
        data, data_skip = data
        batch_out = data_skip.clone()
        x_skip = data_skip.x

        has_innermost = len(data.x) == data.batch.max() + 1

        if precomputed and not has_innermost:
            if not hasattr(data, "up_idx"):
                setattr(batch_out, "up_idx", 0)
            else:
                setattr(batch_out, "up_idx", data.up_idx)

            pre_data = precomputed[batch_out.up_idx]
            batch_out.up_idx = batch_out.up_idx + 1
        else:
            pre_data = None

        if has_innermost:
            x = torch.gather(data.x, 0, data_skip.batch.unsqueeze(-1).repeat((1, data.x.shape[-1])))
        else:
            x = self.upsample_op(data, data_skip, precomputed=pre_data)

        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)

        if hasattr(self, "nn"):
            batch_out.x = self.nn(x)
        else:
            batch_out.x = x
        return batch_out
