import torch.nn as nn
import MinkowskiEngine as ME
from .common import ConvType, NormType

from torch_points3d.utils.config import is_list


class BasicBlock(nn.Module):
    """This module implements a basic residual convolution block using MinkowskiEngine

    Parameters
    ----------
    inplanes: int
        Input dimension
    planes: int
        Output dimension
    dilation: int
        Dilation value
    downsample: nn.Module
        If provided, downsample will be applied on input before doing residual addition
    bn_momentum: float
        Input dimension
    """

    EXPANSION = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1, dimension=-1):
        super(BasicBlock, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension
        )
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension
        )
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    EXPANSION = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1, dimension=-1):
        super(Bottleneck, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(inplanes, planes, kernel_size=1, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)

        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension
        )
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)

        self.conv3 = ME.MinkowskiConvolution(planes, planes * self.EXPANSION, kernel_size=1, dimension=dimension)
        self.norm3 = ME.MinkowskiBatchNorm(planes * self.EXPANSION, momentum=bn_momentum)

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BaseResBlock(nn.Module):
    def __init__(
        self,
        feat_in,
        feat_mid,
        feat_out,
        kernel_sizes=[],
        strides=[],
        dilations=[],
        has_biases=[],
        kernel_generators=[],
        kernel_size=3,
        stride=1,
        dilation=1,
        has_bias=False,
        kernel_generator=None,
        norm_layer=ME.MinkowskiBatchNorm,
        activation=ME.MinkowskiReLU,
        bn_momentum=0.1,
        dimension=-1,
        **kwargs
    ):

        super(BaseResBlock, self).__init__()
        assert dimension > 0

        modules = []

        convolutions_dim = [[feat_in, feat_mid], [feat_mid, feat_mid], [feat_mid, feat_out]]

        kernel_sizes = self.create_arguments_list(kernel_sizes, kernel_size)
        strides = self.create_arguments_list(strides, stride)
        dilations = self.create_arguments_list(dilations, dilation)
        has_biases = self.create_arguments_list(has_biases, has_bias)
        kernel_generators = self.create_arguments_list(kernel_generators, kernel_generator)

        for conv_dim, kernel_size, stride, dilation, has_bias, kernel_generator in zip(
            convolutions_dim, kernel_sizes, strides, dilations, has_biases, kernel_generators
        ):

            modules.append(
                ME.MinkowskiConvolution(
                    conv_dim[0],
                    conv_dim[1],
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    has_bias=has_bias,
                    kernel_generator=kernel_generator,
                    dimension=dimension,
                )
            )

            if norm_layer:
                modules.append(norm_layer(conv_dim[1], momentum=bn_momentum))

            if activation:
                modules.append(activation(inplace=True))

        self.conv = nn.Sequential(*modules)

    @staticmethod
    def create_arguments_list(arg_list, arg):
        if len(arg_list) == 3:
            return arg_list
        return [arg for _ in range(3)]

    def forward(self, x):
        return x, self.conv(x)


class ResnetBlockDown(BaseResBlock):
    def __init__(
        self,
        down_conv_nn=[],
        kernel_sizes=[],
        strides=[],
        dilations=[],
        kernel_size=3,
        stride=1,
        dilation=1,
        norm_layer=ME.MinkowskiBatchNorm,
        activation=ME.MinkowskiReLU,
        bn_momentum=0.1,
        dimension=-1,
        down_stride=2,
        **kwargs
    ):

        super(ResnetBlockDown, self).__init__(
            down_conv_nn[0],
            down_conv_nn[1],
            down_conv_nn[2],
            kernel_sizes=kernel_sizes,
            strides=strides,
            dilations=dilations,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            norm_layer=norm_layer,
            activation=activation,
            bn_momentum=bn_momentum,
            dimension=dimension,
        )

        self.downsample = nn.Sequential(
            ME.MinkowskiConvolution(
                down_conv_nn[0], down_conv_nn[2], kernel_size=2, stride=down_stride, dimension=dimension
            ),
            ME.MinkowskiBatchNorm(down_conv_nn[2]),
        )

    def forward(self, x):

        residual, x = super().forward(x)

        return self.downsample(residual) + x


class ResnetBlockUp(BaseResBlock):
    def __init__(
        self,
        up_conv_nn=[],
        kernel_sizes=[],
        strides=[],
        dilations=[],
        kernel_size=3,
        stride=1,
        dilation=1,
        norm_layer=ME.MinkowskiBatchNorm,
        activation=ME.MinkowskiReLU,
        bn_momentum=0.1,
        dimension=-1,
        up_stride=2,
        skip=True,
        **kwargs
    ):

        self.skip = skip

        super(ResnetBlockUp, self).__init__(
            up_conv_nn[0],
            up_conv_nn[1],
            up_conv_nn[2],
            kernel_sizes=kernel_sizes,
            strides=strides,
            dilations=dilations,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            norm_layer=norm_layer,
            activation=activation,
            bn_momentum=bn_momentum,
            dimension=dimension,
        )

        self.upsample = ME.MinkowskiConvolutionTranspose(
            up_conv_nn[0], up_conv_nn[2], kernel_size=2, stride=up_stride, dimension=dimension
        )

    def forward(self, x, x_skip):
        residual, x = super().forward(x)

        x = self.upsample(residual) + x

        if self.skip:
            return ME.cat(x, x_skip)
        else:
            return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, D=-1):
        # Global coords does not require coords_key
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            ME.MinkowskiLinear(channel, channel // reduction),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(channel // reduction, channel),
            ME.MinkowskiSigmoid(),
        )
        self.pooling = ME.MinkowskiGlobalPooling(dimension=D)
        self.broadcast_mul = ME.MinkowskiBroadcastMultiplication(dimension=D)

    def forward(self, x):
        y = self.pooling(x)
        y = self.fc(y)
        return self.broadcast_mul(x, y)


class SEBasicBlock(BasicBlock):
    def __init__(
        self, inplanes, planes, stride=1, dilation=1, downsample=None, conv_type=ConvType.HYPERCUBE, reduction=16, D=-1
    ):
        super(SEBasicBlock, self).__init__(
            inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, conv_type=conv_type, D=D
        )
        self.se = SELayer(planes, reduction=reduction, D=D)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBasicBlockBN(SEBasicBlock):
    NORM_TYPE = NormType.BATCH_NORM


class SEBasicBlockIN(SEBasicBlock):
    NORM_TYPE = NormType.INSTANCE_NORM


class SEBasicBlockIBN(SEBasicBlock):
    NORM_TYPE = NormType.INSTANCE_BATCH_NORM


class SEBottleneck(Bottleneck):
    def __init__(
        self, inplanes, planes, stride=1, dilation=1, downsample=None, conv_type=ConvType.HYPERCUBE, D=3, reduction=16
    ):
        super(SEBottleneck, self).__init__(
            inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, conv_type=conv_type, D=D
        )
        self.se = SELayer(planes * self.expansion, reduction=reduction, D=D)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneckBN(SEBottleneck):
    NORM_TYPE = NormType.BATCH_NORM


class SEBottleneckIN(SEBottleneck):
    NORM_TYPE = NormType.INSTANCE_NORM


class SEBottleneckIBN(SEBottleneck):
    NORM_TYPE = NormType.INSTANCE_BATCH_NORM
