import torch.nn as nn
import MinkowskiEngine as ME

from src.utils.config import is_list


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
                modules.append(norm_layer(conv_dim[0], momentum=bn_momentum))

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
        down_stride=1,
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
                down_conv_nn[0], down_conv_nn[2], kernel_size=1, stride=down_stride, dimension=dimension
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
        up_stride=1,
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
            up_conv_nn[0], up_conv_nn[2], kernel_size=1, stride=up_stride, dimension=dimension
        )

    def forward(self, x, x_skip):

        residual, x = super().forward(x)

        x = self.upsample(residual + x)

        if self.skip:
            return torch.cat([x, x_skip], -1)
        else:
            return x
