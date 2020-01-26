import torch
from torch_geometric.data import Batch

from .kernels import KPConvLayer
from src.core.neighbourfinder import RadiusNeighbourFinder
from src.core.data_transform import GridSampling
from src.utils.enums import ConvolutionFormat


class SimpleBlock(torch.nn.Module):
    """
    simple layer with KPConv convolution -> activation -> BN
    we can perform a stride version (just change the query and the neighbors)
    """

    CONV_TYPE = ConvolutionFormat.PARTIAL_DENSE.value[-1]

    def __init__(
        self,
        down_conv_nn=None,
        radius=None,
        kp_extent=1,
        density_parameter=1,
        max_num_neighbors=64,
        activation=torch.nn.LeakyReLU(negative_slope=0.2),
        bn_momentum=0.1,
        bn=torch.nn.BatchNorm1d,
        grid_size=None,
        **kwargs
    ):
        super(SimpleBlock, self).__init__()
        assert len(down_conv_nn) == 2
        num_inputs, num_outputs = down_conv_nn
        self.kp_conv = KPConvLayer(
            num_inputs, num_outputs, radius=radius, kp_extent=kp_extent, density_parameter=density_parameter
        )
        if bn:
            self.bn = bn(num_outputs, momentum=bn_momentum)
        else:
            self.bn = None
        self.activation = activation

        self.neighbour_finder = RadiusNeighbourFinder(radius, max_num_neighbors, conv_type=self.CONV_TYPE)
        if grid_size:
            self.grid_size = grid_size
            self.sampler = GridSampling(grid_size)

    def forward(self, data):
        if self.sampler:
            querry_data = self.sampler(data.clone())
        else:
            querry_data = data.clone()

        q_pos, q_batch = querry_data.pos, querry_data.batch
        idx_neighbour, _ = self.neighbour_finder(q_pos, data.pos, batch_x=q_batch, batch_y=data.batch)
        x = self.kp_conv(q_pos, data.pos, idx_neighbour, data.x,)
        if self.bn:
            x = self.bn(x)
        x = self.activation(x)

        querry_data.x = x
        return querry_data

    def extra_repr(self):
        return str(self.sampler) + "," + str(self.neighbour_finder)
