import MinkowskiEngine as ME

from .common import NormType, get_norm
from torch_points3d.core.common_modules import Seq, Identity


class ResBlock(ME.MinkowskiNetwork):
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
                    has_bias=False,
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
                    has_bias=False,
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
                        has_bias=False,
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


class SRBlockDown(ME.MinkowskiNetwork):
    """
    Resnet block that looks like

    in --- strided conv --- 3x3 - BN - RELU (2 times) --- sum --[... N times]
                         |                                  |
                         |--------------- 1x1 --------------|
    """

    CONVOLUTION = ME.MinkowskiConvolution

    def __init__(self, down_conv_nn=[], kernel_size=2, dilation=1, dimension=3, stride=2, N=1, **kwargs):
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
                    has_bias=False,
                    dimension=dimension,
                )
            )
            .append(ME.MinkowskiBatchNorm(conv1_output))
            .append(ME.MinkowskiReLU())
        )

        if N > 0:
            self.blocks = Seq()
            for _ in range(N):
                self.blocks.append(ResBlock(conv1_output, down_conv_nn[1], self.CONVOLUTION, dimension=dimension))
                conv1_output = down_conv_nn[1]
        else:
            self.blocks = None

    def forward(self, x):
        out = self.conv_in(x)
        if self.blocks:
            out = self.blocks(out)
        return out


class SRBlockUp(SRBlockDown):
    """
    block for unwrapped Resnet
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
