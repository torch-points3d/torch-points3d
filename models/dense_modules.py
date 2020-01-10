import torch
import torch_points as tp
from torch_geometric.data import Data
import etw_pytorch_utils as pt_utils

from .core_sampling_and_search import BaseMSNeighbourFinder
from .core_modules import BaseConvolution


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

    def conv(self, x, pos, new_pos, radius_idx, scale_idx):
        """ Implements a Dense convolution where radius_idx represents
        the indexes of the points in x and pos to be agragated into the new feature
        for each point in new_pos

        Arguments:
            x -- Previous features [B, C, N]
            pos -- Previous positions [B, N, 3]
            new_pos  -- Sampled positions [B, npoints, 3]
            radius_idx -- Indexes to group [B, npoints, nsample]
            scale_idx -- Scale index in multiscale convolutional layers
        """
        raise NotImplementedError

    def forward(self, pos, x=None):
        """
        Arguments:
            x -- Previous features [B, C, N]
            pos -- Previous positions [B, N, 3]
        """
        idx = self.sampler(pos)
        pos_flipped = pos.transpose(1, 2).contiguous()
        new_pos = tp.gather_operation(pos_flipped, idx).transpose(1, 2).contiguous()

        ms_x = []
        for scale_idx in range(self.neighbour_finder.num_scales):
            if self._precompute_multi_scale:
                raise NotImplementedError()
            else:
                radius_idx = self.neighbour_finder(pos, new_pos, scale_idx=scale_idx)
            ms_x.append(self.conv(x, pos, new_pos, radius_idx, scale_idx))
        new_x = torch.cat(ms_x, 1)
        return new_pos, new_x


class BaseDenseConvolutionUp(BaseConvolution):
    def __init__(self, neighbour_finder, *args, **kwargs):
        super(BaseDenseConvolutionUp, self).__init__(None, neighbour_finder, *args, **kwargs)

        self._precompute_multi_scale = kwargs.get("precompute_multi_scale", None)
        self._index = kwargs.get("index", None)
        self._skip = kwargs.get("skip", True)

    def conv(self, x, pos, pos_skip):
        raise NotImplementedError

    def forward(self, x, x_skip, pos, pos_skip):
        new_features = self.conv(x, pos, pos_skip)
        if x_skip is not None:
            new_features = torch.cat([new_features, x_skip], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = new_features

        new_features = new_features.unsqueeze(-1)
        if hasattr(self, "nn"):
            new_features = self.nn(new_features)
        else:
            new_features = new_features
        return new_features.squeeze(-1)


class DenseFPModule(BaseDenseConvolutionUp):
    def __init__(self, up_conv_nn, **kwargs):
        super(DenseFPModule, self).__init__(None)
        self.nn = pt_utils.SharedMLP(up_conv_nn)

    def conv(self, x, pos, pos_skip):
        if pos is not None:
            dist, idx = tp.three_nn(pos_skip, pos)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = tp.three_interpolate(x, idx, weight)
        else:
            interpolated_feats = x.expand(*(x.size()[0:2] + [pos.size(1)]))
        return interpolated_feats
