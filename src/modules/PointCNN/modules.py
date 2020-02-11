from math import ceil

import torch
from torch.nn import Sequential as S, Linear as L, BatchNorm1d as BN
from torch.nn import ELU, Conv1d
from torch_geometric.nn import Reshape
from torch_geometric.nn.inits import reset

from src.core.spatial_ops import RandomSampler, FPSSampler, DilatedKNNNeighbourFinder
from src.core.base_conv.message_passing import *


# XConv from torch geometric, modified for this framework
# https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/conv/x_conv.py


class XConv(torch.nn.Module):
    r"""The convolutional operator on :math:`\mathcal{X}`-transformed points
    from the `"PointCNN: Convolution On X-Transformed Points"
    <https://arxiv.org/abs/1801.07791>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathrm{Conv}\left(\mathbf{K},
        \gamma_{\mathbf{\Theta}}(\mathbf{P}_i - \mathbf{p}_i) \times
        \left( h_\mathbf{\Theta}(\mathbf{P}_i - \mathbf{p}_i) \, \Vert \,
        \mathbf{x}_i \right) \right),
    where :math:`\mathbf{K}` and :math:`\mathbf{P}_i` denote the trainable
    filter and neighboring point positions of :math:`\mathbf{x}_i`,
    respectively.
    :math:`\gamma_{\mathbf{\Theta}}` and :math:`h_{\mathbf{\Theta}}` describe
    neural networks, *i.e.* MLPs, where :math:`h_{\mathbf{\Theta}}`
    individually lifts each point into a higher-dimensional space, and
    :math:`\gamma_{\mathbf{\Theta}}` computes the :math:`\mathcal{X}`-
    transformation matrix based on *all* points in a neighborhood.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        dim (int): Point cloud dimensionality.
        kernel_size (int): Size of the convolving kernel, *i.e.* number of
            neighbors including self-loops.
        hidden_channels (int, optional): Output size of
            :math:`h_{\mathbf{\Theta}}`, *i.e.* dimensionality of lifted
            points. If set to :obj:`None`, will be automatically set to
            :obj:`in_channels / 4`. (default: :obj:`None`)
        dilation (int, optional): The factor by which the neighborhood is
            extended, from which :obj:`kernel_size` neighbors are then
            uniformly sampled. Can be interpreted as the dilation rate of
            classical convolutional operators. (default: :obj:`1`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_cluster.knn_graph`.
    """

    def __init__(
        self, in_channels, out_channels, dim, kernel_size, hidden_channels=None, dilation=1, bias=True, **kwargs,
    ):
        super(XConv, self).__init__()

        self.in_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels // 4
        assert hidden_channels > 0
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.kwargs = kwargs

        C_in, C_delta, C_out = in_channels, hidden_channels, out_channels
        D, K = dim, kernel_size

        self.mlp1 = S(
            L(dim, C_delta), ELU(), BN(C_delta), L(C_delta, C_delta), ELU(), BN(C_delta), Reshape(-1, K, C_delta),
        )

        self.mlp2 = S(
            L(D * K, K ** 2),
            ELU(),
            BN(K ** 2),
            Reshape(-1, K, K),
            Conv1d(K, K ** 2, K, groups=K),
            ELU(),
            BN(K ** 2),
            Reshape(-1, K, K),
            Conv1d(K, K ** 2, K, groups=K),
            BN(K ** 2),
            Reshape(-1, K, K),
        )

        C_in = C_in + C_delta
        depth_multiplier = int(ceil(C_out / C_in))
        self.conv = S(
            Conv1d(C_in, C_in * depth_multiplier, K, groups=C_in),
            Reshape(-1, C_in * depth_multiplier),
            L(C_in * depth_multiplier, C_out, bias=bias),
        )

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp1)
        reset(self.mlp2)
        reset(self.conv)

    def forward(self, x, pos, edge_index):

        # posTo = the points that will be centers of convolutions
        # posFrom = points that have edges to the centers of convolutions
        # For a down conv, posFrom = pos, posTo = pos[idx]
        # For an up conv, posFrom = pos, posTo = pos_skip
        posFrom, posTo = pos

        (N, D), K = posTo.size(), self.kernel_size

        idxFrom, idxTo = edge_index

        relPos = posTo[idxTo] - posFrom[idxFrom]

        x_star = self.mlp1(relPos)
        # x_star = self.mlp1(relPos.view(len(row), D))
        if x is not None:
            x = x.unsqueeze(-1) if x.dim() == 1 else x
            x = x[idxFrom].view(N, K, self.in_channels)
            x_star = torch.cat([x_star, x], dim=-1)
        x_star = x_star.transpose(1, 2).contiguous()
        x_star = x_star.view(N, self.in_channels + self.hidden_channels, K, 1)

        transform_matrix = self.mlp2(relPos.view(N, K * D))
        transform_matrix = transform_matrix.view(N, 1, K, K)

        x_transformed = torch.matmul(transform_matrix, x_star)
        x_transformed = x_transformed.view(N, -1, K)

        out = self.conv(x_transformed)

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class PointCNNConvDown(BaseConvolutionDown):
    def __init__(
        self, inN=None, outN=None, K=None, D=None, C1=None, C2=None, hidden_channel=None, *args, **kwargs,
    ):
        super(PointCNNConvDown, self).__init__(FPSSampler(outN / inN), DilatedKNNNeighbourFinder(K, D))

        self._conv = XConv(C1, C2, 3, K, hidden_channels=hidden_channel)

    def conv(self, x, pos, edge_index, batch):
        return self._conv.forward(x, pos, edge_index)


class PointCNNConvUp(BaseConvolutionUp):
    def __init__(self, K=None, D=None, C1=None, C2=None, *args, **kwargs):
        super(PointCNNConvUp, self).__init__(DilatedKNNNeighbourFinder(K, D))

        self._conv = XConv(C1, C2, 3, K)

    def conv(self, x, pos, pos_skip, batch, batch_skip, edge_index):
        return self._conv.forward(x, (pos, pos_skip), edge_index)
