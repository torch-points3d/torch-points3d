import torch.nn as nn
import MinkowskiEngine as ME
from .common import ConvType, NormType

from torch_points3d.utils.config import is_list


class OuterEquivariantMap(nn.Module):
    def __init__(self, input_nc=3, output_nc=64, kernel_size=[16, 16, 16], stride=16, dilation=1, size_conv=3, D=3):

        super(OuterEquivariantMap, self).__init__()

        self.pool = ME.MinkowskiAvgPooling(kernel_size=[16, 16, 16], stride=16, dilation=dilation, dimension=D)
        self.unpool = ME.MinkowskiPoolingTranspose(kernel_size=[16, 16, 16], stride=16, dilation=dilation, dimension=D)
        self.conv3d = ME.MinkowskiConvolution(
            input_nc, output_nc, kernel_size=size_conv, stride=1, dilation=dilation, dimension=D
        )

    def forward(self, x):
        out = self.unpool(self.conv3d(self.pool(x)))
        return out


class InnerEquivariantMap(nn.Module):
    """
    TODO: add attention mechanism
    """

    def __init__(self, input_nc, output_nc, L=10, D=3):
        super(InnerEquivariantMap, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(input_nc, output_nc, kernel_size=1, stride=1, dimension=D)

    def forward(self, x):
        return self.conv1(x)


class EMLayer(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        L=10,
        kernel_size=[16, 16, 16],
        stride=16,
        size_conv=3,
        dilation=1,
        bn_momentum=0.02,
        D=3,
    ):
        super(EMLayer, self).__init__()
        self.oem = OuterEquivariantMap(input_nc, output_nc, kernel_size, stride, dilation, size_conv, D)
        self.iem = InnerEquivariantMap(input_nc, output_nc, L, D)
        self.norm = ME.MinkowskiBatchNorm(output_nc, momentum=bn_momentum)
        self.activation = ME.MinkowskiReLU(inplace=False)

    def forward(self, x):
        return self.activation(self.norm(self.oem(x) + self.iem(x)))


class ResEMBlock(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        L=10,
        kernel_size=[16, 16, 16],
        stride=16,
        dilation=1,
        size_conv=3,
        bn_momentum=0.02,
        D=3,
    ):
        super(ResEMBlock, self).__init__()
        self.em1 = EMLayer(input_nc, output_nc, L, kernel_size, stride, dilation, size_conv, bn_momentum, D)
        self.em2 = EMLayer(output_nc, output_nc, L, kernel_size, stride, dilation, size_conv, bn_momentum, D)

    def forward(self, x):
        residual = x
        out = self.em2(self.em1(x))
        out += residual
        return out


class EquivariantMapNetwork(nn.Module):
    def __init__(self, input_nc=3, dim_feat=64, output_nc=32, num_layer=20, kernel_size=16, stride=16, dilation=1, D=3):
        super(EquivariantMapNetwork, self).__init__()
        self.layer1 = EMLayer(input_nc, dim_feat, kernel_size=kernel_size, stride=stride, dilation=dilation, D=D)
        self.list_res = nn.ModuleList()
        for _ in range(num_layer):
            self.list_res.append(
                ResEMBlock(dim_feat, dim_feat, kernel_size=kernel_size, stride=stride, dilation=dilation, D=D)
            )

        self.last = ME.MinkowskiConvolution(dim_feat, output_nc, kernel_size=1, stride=1, dimension=D)

    def forward(self, x):
        x = self.layer1(x)
        for i in range(len(self.list_res)):
            x = self.list_res[i](x)
        return self.last(x)
