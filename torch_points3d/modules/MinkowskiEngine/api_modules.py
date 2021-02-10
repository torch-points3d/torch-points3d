import torch
import MinkowskiEngine as ME
import sys

from .common import NormType, get_norm
from torch_points3d.core.common_modules import Seq, Identity


class ResBlock(ME.MinkowskiNetwork):
    """
    Basic ResNet type block

    Parameters
    ----------
    input_nc:
        Number of input channels
    output_nc:
        number of output channels
    convolution
        Either MinkowskConvolution or MinkowskiConvolutionTranspose
    dimension:
        Dimension of the spatial grid
    """

    def __init__(self, input_nc, output_nc, convolution, dimension=3):
        ME.MinkowskiNetwork.__init__(self, dimension)
        self.block = (
            Seq()
            .append(
                convolution(
                    in_channels=input_nc,
                    out_channels=output_nc,
                    kernel_size=3,
                    stride=1,
                    dilation=1,
                    bias=False,
                    dimension=dimension,
                )
            )
            .append(ME.MinkowskiBatchNorm(output_nc))
            .append(ME.MinkowskiReLU())
            .append(
                convolution(
                    in_channels=output_nc,
                    out_channels=output_nc,
                    kernel_size=3,
                    stride=1,
                    dilation=1,
                    bias=False,
                    dimension=dimension,
                )
            )
            .append(ME.MinkowskiBatchNorm(output_nc))
            .append(ME.MinkowskiReLU())
        )

        if input_nc != output_nc:
            self.downsample = (
                Seq()
                .append(
                    convolution(
                        in_channels=input_nc,
                        out_channels=output_nc,
                        kernel_size=1,
                        stride=1,
                        dilation=1,
                        bias=False,
                        dimension=dimension,
                    )
                )
                .append(ME.MinkowskiBatchNorm(output_nc))
            )
        else:
            self.downsample = None

    def forward(self, x):
        out = self.block(x)
        if self.downsample:
            out += self.downsample(x)
        else:
            out += x
        return out


class BottleneckBlock(ME.MinkowskiNetwork):
    """
    Bottleneck block with residual
    """

    def __init__(self, input_nc, output_nc, convolution, dimension=3, reduction=4):
        self.block = (
            Seq()
            .append(
                convolution(
                    in_channels=input_nc,
                    out_channels=output_nc // reduction,
                    kernel_size=1,
                    stride=1,
                    dilation=1,
                    bias=False,
                    dimension=dimension,
                )
            )
            .append(ME.MinkowskiBatchNorm(output_nc // reduction))
            .append(ME.MinkowskiReLU())
            .append(
                convolution(
                    output_nc // reduction,
                    output_nc // reduction,
                    kernel_size=3,
                    stride=1,
                    dilation=1,
                    bias=False,
                    dimension=dimension,
                )
            )
            .append(ME.MinkowskiBatchNorm(output_nc // reduction))
            .append(ME.MinkowskiReLU())
            .append(
                convolution(
                    output_nc // reduction,
                    output_nc,
                    kernel_size=1,
                    stride=1,
                    dilation=1,
                    bias=False,
                    dimension=dimension,
                )
            )
            .append(ME.MinkowskiBatchNorm(output_nc))
            .append(ME.MinkowskiReLU())
        )

        if input_nc != output_nc:
            self.downsample = (
                Seq()
                .append(
                    convolution(
                        in_channels=input_nc,
                        out_channels=output_nc,
                        kernel_size=1,
                        stride=1,
                        dilation=1,
                        bias=False,
                        dimension=dimension,
                    )
                )
                .append(ME.MinkowskiBatchNorm(output_nc))
            )
        else:
            self.downsample = None

    def forward(self, x):
        out = self.block(x)
        if self.downsample:
            out += self.downsample(x)
        else:
            out += x
        return out


class SELayer(torch.nn.Module):
    """
    Squeeze and excite layer

    Parameters
    ----------
    channel:
        size of the input and output
    reduction:
        magnitude of the compression
    D:
        dimension of the kernels
    """

    def __init__(self, channel, reduction=16, dimension=3):
        # Global coords does not require coords_key
        super(SELayer, self).__init__()
        self.fc = torch.nn.Sequential(
            ME.MinkowskiLinear(channel, channel // reduction),
            ME.MinkowskiReLU(),
            ME.MinkowskiLinear(channel // reduction, channel),
            ME.MinkowskiSigmoid(),
        )
        self.pooling = ME.MinkowskiGlobalPooling()
        self.broadcast_mul = ME.MinkowskiBroadcastMultiplication()

    def forward(self, x):
        y = self.pooling(x)
        y = self.fc(y)
        return self.broadcast_mul(x, y)


class SEBlock(ResBlock):
    """
    ResBlock with SE layer
    """

    def __init__(self, input_nc, output_nc, convolution, dimension=3, reduction=16):
        super().__init__(input_nc, output_nc, convolution, dimension=3)
        self.SE = SELayer(output_nc, reduction=reduction, dimension=dimension)

    def forward(self, x):
        out = self.block(x)
        out = self.SE(out)
        if self.downsample:
            out += self.downsample(x)
        else:
            out += x
        return out


class SEBottleneckBlock(BottleneckBlock):
    """
    BottleneckBlock with SE layer
    """

    def __init__(self, input_nc, output_nc, convolution, dimension=3, reduction=16):
        super().__init__(input_nc, output_nc, convolution, dimension=3, reduction=4)
        self.SE = SELayer(output_nc, reduction=reduction, dimension=dimension)

    def forward(self, x):
        out = self.block(x)
        out = self.SE(out)
        if self.downsample:
            out += self.downsample(x)
        else:
            out += x
        return out


_res_blocks = sys.modules[__name__]


class ResNetDown(ME.MinkowskiNetwork):
    """
    Resnet block that looks like

    in --- strided conv ---- Block ---- sum --[... N times]
                         |              |
                         |-- 1x1 - BN --|
    """

    CONVOLUTION = ME.MinkowskiConvolution

    def __init__(
        self, down_conv_nn=[], kernel_size=2, dilation=1, dimension=3, stride=2, N=1, block="ResBlock", **kwargs
    ):
        block = getattr(_res_blocks, block)
        ME.MinkowskiNetwork.__init__(self, dimension)
        if stride > 1:
            conv1_output = down_conv_nn[0]
        else:
            conv1_output = down_conv_nn[1]

        self.conv_in = (
            Seq()
            .append(
                self.CONVOLUTION(
                    in_channels=down_conv_nn[0],
                    out_channels=conv1_output,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    bias=False,
                    dimension=dimension,
                )
            )
            .append(ME.MinkowskiBatchNorm(conv1_output))
            .append(ME.MinkowskiReLU())
        )

        if N > 0:
            self.blocks = Seq()
            for _ in range(N):
                self.blocks.append(block(conv1_output, down_conv_nn[1], self.CONVOLUTION, dimension=dimension))
                conv1_output = down_conv_nn[1]
        else:
            self.blocks = None

    def forward(self, x):
        out = self.conv_in(x)
        if self.blocks:
            out = self.blocks(out)
        return out


class ResNetUp(ResNetDown):
    """
    Same as Down conv but for the Decoder
    """

    CONVOLUTION = ME.MinkowskiConvolutionTranspose

    def __init__(self, up_conv_nn=[], kernel_size=2, dilation=1, dimension=3, stride=2, N=1, **kwargs):
        super().__init__(
            down_conv_nn=up_conv_nn,
            kernel_size=kernel_size,
            dilation=dilation,
            dimension=dimension,
            stride=stride,
            N=N,
            **kwargs
        )

    def forward(self, x, skip):
        if skip is not None:
            inp = ME.cat(x, skip)
        else:
            inp = x
        return super().forward(inp)
