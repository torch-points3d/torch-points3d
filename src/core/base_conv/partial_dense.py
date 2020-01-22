from abc import ABC, abstractmethod
from typing import *
import math
from functools import partial
from typing import Dict, Any
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import (
    Sequential as Seq,
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
from torch_geometric.utils import scatter_
import torch_points as tp
import etw_pytorch_utils as pt_utils

from src.core.neighbourfinder import *
from .base_conv import BaseConvolution
from src.core.common_modules import MLP


#################### THOSE MODULES IMPLEMENTS THE BASE PARTIAL_DENSE CONV API ############################


def copy_from_to(data, batch):
    for key in data.keys:
        if key not in batch.keys:
            setattr(batch, key, getattr(data, key, None))


class BasePartialDenseConvolutionDown(BaseConvolution):

    CONV_TYPE = ConvolutionFormat.PARTIAL_DENSE.value[-1]

    def __init__(self, sampler, neighbour_finder, *args, **kwargs):
        super(BasePartialDenseConvolutionDown, self).__init__(sampler, neighbour_finder, *args, **kwargs)

        self._precompute_multi_scale = kwargs.get("precompute_multi_scale", None)
        self._index = kwargs.get("index", None)

        assert self.CONV_TYPE == kwargs.get(
            "conv_type", None
        ), "The conv_type shoud be the same as the one used to defined the convolution {}".format(self.CONV_TYPE)

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

    def forward(self, data):
        batch_obj = Batch()
        x, pos, batch = data.x, data.pos, data.batch
        idx_sampler = self.sampler(pos=pos, x=x, batch=batch)

        idx_neighbour, _ = self.neighbour_finder(pos, pos, batch_x=batch, batch_y=batch)

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

    def forward(self, data):
        batch_obj = Batch()
        x, pos, batch = data.x, data.pos, data.batch
        x = self.nn(torch.cat([x, pos], dim=1))
        x = self.pool(x, batch)
        batch_obj.x = x
        batch_obj.pos = pos.new_zeros((x.size(0), 3))
        batch_obj.batch = torch.arange(x.size(0), device=x.device)
        copy_from_to(data, batch_obj)
        return batch_obj
