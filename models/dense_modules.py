from torch_geometric.data import Data
import torch_points as tp
import torch
import etw_pytorch_utils as pt_utils

from models.core_modules import BaseConvolution
from models.core_sampling_and_search import BaseMSNeighbourFinder


def _copy_from_to(data, batch):
    for key in data.keys:
        if key not in batch.keys:
            setattr(batch, key, getattr(data, key, None))


class BaseDenseConvolutionDown(BaseConvolution):
    """ Multiscale convolution down (also supports single scale).
        Convolution kernel is shared accross the scales

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
        Returns
            Features -- [N, npoints, output_channels]
        """
        raise NotImplementedError

    def forward(self, data):
        batch_obj = Data()
        x, pos = data.x, data.pos
        if self._precompute_multi_scale:
            idx = getattr(data, "idx_{}".format(self._index), None)
        else:
            idx = self.sampler(pos)
            batch_obj.idx = idx

        pos_flipped = pos.transpose(1, 2).contiguous()  # [B, 3, N]
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
        _copy_from_to(data, batch_obj)
        return batch_obj


class GlobalDenseBaseModule(torch.nn.Module):
    def __init__(self, nn, aggr='max'):
        super(GlobalDenseBaseModule, self).__init__()
        self.nn = pt_utils.SharedMLP(nn)

    def forward(self, data):
        batch_obj = Data()
        x, pos = data.x, data.pos
        pos_flipped = pos.transpose(1, 2).contiguous()
        x = self.nn(torch.cat([x, pos_flipped], dim=1).unsqueeze(-1))
        x = x.squeeze().max(-1)[0]
        batch_obj.x = x
        batch_obj.pos = pos.new_zeros((x.size(0), 3, 1))
        _copy_from_to(data, batch_obj)
        return batch_obj


class BaseDenseConvolutionUp(BaseConvolution):
    def __init__(self, neighbour_finder, *args, **kwargs):
        super(BaseDenseConvolutionUp, self).__init__(None, neighbour_finder, *args, **kwargs)

        self._precompute_multi_scale = kwargs.get("precompute_multi_scale", None)
        self._index = kwargs.get("index", None)
        self._skip = kwargs.get("skip", True)

    def conv(self, x, x_skip, pos, pos_skip):
        raise NotImplementedError

    def forward(self, data):
        batch_obj = Data()
        data, data_skip = data
        x, pos = data.x.transpose(2, 1).contiguous(), data.pos
        x_skip, pos_skip = data_skip.x, data_skip.pos
        x = self.conv(x, x_skip, pos, pos_skip)

        if x_skip is not None and self._skip:
            if x.shape[-1] == x_skip.shape[-1]:
                x = torch.cat([x, x_skip], dim=1)
            else:
                x_skip = x_skip.transpose(1, 2).contiguous()
                x = torch.cat([x, x_skip], dim=1)

        x = x.unsqueeze(-1)
        if hasattr(self, 'nn'):
            x = self.nn(x)
        else:
            x = x
        batch_obj.x = x.transpose(2, 1).contiguous().squeeze(-1)
        _copy_from_to(data_skip, batch_obj)
        return batch_obj


class DenseFPModule(BaseDenseConvolutionUp):
    def __init__(self, up_conv_nn, nb_feature=None, **kwargs):
        super(DenseFPModule, self).__init__(None)

        self.nn = pt_utils.SharedMLP(up_conv_nn)

    def conv(self, x, x_skip, pos, pos_skip):
        dist, idx = tp.three_nn(pos_skip, pos)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm

        x = x.squeeze(-1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(-1)

        # print(x.shape, x_skip.shape, pos.shape, pos_skip.shape)
        interpolated_feats = tp.three_interpolate(x, idx, weight)
        return interpolated_feats
