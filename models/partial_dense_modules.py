import torch
from torch_geometric.data import Batch
from .core_modules import BaseConvolution, copy_from_to
from .core_sampling_and_search import BaseMSNeighbourFinder
from utils_folder.enums import ConvolutionFormat


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
        pos_centered_neighbour = pos[idx_neighbour] - pos[:-1].unsqueeze(1)  # Centered the points

        batch_obj.x = self.conv(x, pos, x_neighbour, pos_centered_neighbour, idx_neighbour, idx_sampler)

        batch_obj.pos = pos[idx_sampler]
        batch_obj.batch = batch[idx_sampler]
        copy_from_to(data, batch_obj)
        return batch_obj
