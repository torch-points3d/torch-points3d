import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, BatchNorm2d
import torch_points as tp
from models.core_modules import *
from models.core_sampling_and_search import DenseRadiusNeighbourFinder, DenseFPSSampler


class SADenseModule(BaseDenseConvolutionDown):
    def __init__(self, ratio=None, radius=None, radius_num_point=None, down_conv_nn=None, nb_feature=None, *args, **kwargs):
        super(SADenseModule, self).__init__(DenseFPSSampler(ratio=ratio),
                                            DenseRadiusNeighbourFinder(radius, max_num_neighbors=radius_num_point), *args, **kwargs)

        self._local_nn = SharedMLP(down_conv_nn, bn=True) if down_conv_nn is not None else None
        self._dim_in = down_conv_nn[0] if down_conv_nn is not None else None

        self._radius = radius
        self._ratio = ratio
        self._num_points = radius_num_point

    def _prepare_features(self, x, pos, new_pos, radius_idx):
        pos_trans = pos.transpose(1, 2).contiguous()
        grouped_pos = tp.grouping_operation(pos_trans, radius_idx)  # (B, 3, npoint, nsample)
        grouped_pos -= new_pos.transpose(1, 2).unsqueeze(-1)

        if x is not None:
            x_trans = x.view((pos.shape[0], self._dim_in - pos.shape[-1], -1)).contiguous()
            grouped_features = tp.grouping_operation(x_trans, radius_idx)
            new_features = torch.cat(
                [grouped_pos, grouped_features], dim=1
            )  # (B, C + 3, npoint, nsample)
        else:
            new_features = grouped_pos

        return new_features

    def conv(self, x, pos, new_pos, radius_idx):
        features = self._prepare_features(x, pos, new_pos, radius_idx)
        new_features = self._local_nn(features)
        new_features = F.max_pool2d(
            new_features, kernel_size=[1, new_features.size(3)]
        )  # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
        return new_features

    def extra_repr(self):
        return '{}(ratio {}, radius {}, radius_points {})'.format(self.__class__.__name__, self._ratio, self._radius, self._num_points)
