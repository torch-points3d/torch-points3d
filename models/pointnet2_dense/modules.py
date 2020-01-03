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

        self._radius = radius
        self._ratio = ratio
        self._num_points = radius_num_point

    def _prepare_features(self, x, pos, new_pos, radius_idx):
        pos_trans = pos.transpose(1, 2).contiguous()
        grouped_pos = tp.grouping_operation(pos_trans, radius_idx)  # (B, 3, npoint, nsample)
        grouped_pos -= new_pos.transpose(1, 2).unsqueeze(-1)

        if x is not None:
            grouped_features = tp.grouping_operation(x, radius_idx)
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


class SharedMLP(nn.Sequential):
    def __init__(
        self,
        layers,
        bn=False,
        activation=nn.ReLU(inplace=True),
        preact=False,
        first=False,
        name="",
    ):
        # type: (SharedMLP, List[int], bool, Any, bool, bool, AnyStr) -> None
        super(SharedMLP, self).__init__()

        for i in range(len(layers) - 1):
            self.add_module(
                name + "layer{}".format(i),
                Conv2d(
                    layers[i],
                    layers[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0))
                    else None,
                    preact=preact,
                ),
            )


class _ConvBase(nn.Sequential):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size,
        stride,
        padding,
        dilation,
        activation,
        bn,
        init,
        conv=None,
        norm_layer=None,
        bias=True,
        preact=False,
        name="",
    ):
        super(_ConvBase, self).__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = norm_layer(out_size)
            else:
                bn_unit = norm_layer(in_size)

        if preact:
            if bn:
                self.add_module(name + "normlayer", bn_unit)

            if activation is not None:
                self.add_module(name + "activation", activation)

        self.add_module(name + "conv", conv_unit)

        if not preact:
            if bn:
                self.add_module(name + "normlayer", bn_unit)

            if activation is not None:
                self.add_module(name + "activation", activation)


class Conv1d(_ConvBase):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=nn.init.kaiming_normal_,
        bias=True,
        preact=False,
        name="",
        norm_layer=BatchNorm1d,
    ):
        # type: (Conv1d, int, int, int, int, int, int, Any, bool, Any, bool, bool, AnyStr, _BNBase) -> None
        super(Conv1d, self).__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            dilation,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            norm_layer=norm_layer,
            bias=bias,
            preact=preact,
            name=name,
        )


class Conv2d(_ConvBase):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=nn.init.kaiming_normal_,
        bias=True,
        preact=False,
        name="",
        norm_layer=BatchNorm2d,
    ):
        # type: (Conv2d, int, int, Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Any, bool, Any, bool, bool, AnyStr, _BNBase) -> None
        super(Conv2d, self).__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            dilation,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            norm_layer=norm_layer,
            bias=bias,
            preact=preact,
            name=name,
        )
