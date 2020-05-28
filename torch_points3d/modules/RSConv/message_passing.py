import torch
from torch.nn import ReLU
from torch_geometric.nn import MessagePassing


from torch_points3d.core.base_conv.message_passing import *
from torch_points3d.core.spatial_ops import *


class Convolution(MessagePassing):
    r"""The Relation Shape Convolution layer from "Relation-Shape Convolutional Neural Network for Point Cloud Analysis"
    https://arxiv.org/pdf/1904.07601

    local_nn - an MLP which is applied to the relation vector h_ij between points i and j to determine
    the weights applied to each element of the feature for x_j

    global_nn - an optional MPL for channel-raising following the convolution

    """

    def __init__(self, local_nn, activation=ReLU(), global_nn=None, aggr="max", **kwargs):
        super(Convolution, self).__init__(aggr=aggr)
        self.local_nn = MLP(local_nn)
        self.activation = activation
        self.global_nn = MLP(global_nn) if global_nn is not None else None

    def forward(self, x, pos, edge_index):
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, pos_i, pos_j, x_j):

        if x_j is None:
            x_j = pos_j

        vij = pos_i - pos_j
        dij = torch.norm(vij, dim=1).unsqueeze(1)

        hij = torch.cat([dij, vij, pos_i, pos_j,], dim=1)

        M_hij = self.local_nn(hij)

        msg = M_hij * x_j

        return msg

    def update(self, aggr_out):
        x = self.activation(aggr_out)
        if self.global_nn is not None:
            x = self.global_nn(x)
        return x


class RSConvDown(BaseConvolutionDown):
    def __init__(self, ratio=None, radius=None, local_nn=None, down_conv_nn=None, *args, **kwargs):
        super(RSConvDown, self).__init__(FPSSampler(ratio), RadiusNeighbourFinder(radius), *args, **kwargs)

        self._conv = Convolution(local_nn=local_nn, global_nn=down_conv_nn)

    def conv(self, x, pos, edge_index, batch):
        return self._conv(x, pos, edge_index)
