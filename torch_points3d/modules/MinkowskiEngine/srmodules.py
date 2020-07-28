import MinkowskiEngine as ME

from .common import NormType, get_norm
from torch_points3d.core.common_modules import Seq, Identity


class SRBlockDown(ME.MinkowskiNetwork):
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
            self.block = Seq()
            for _ in range(N):
                self.block.append(
                    Seq()
                    .append(
                        self.CONVOLUTION(
                            in_channels=conv1_output,
                            out_channels=down_conv_nn[1],
                            kernel_size=2,
                            stride=1,
                            dilation=dilation,
                            has_bias=False,
                            dimension=dimension,
                        )
                    )
                    .append(ME.MinkowskiBatchNorm(down_conv_nn[1]))
                    .append(ME.MinkowskiReLU())
                )
                conv1_output = down_conv_nn[1]
        else:
            self.block = None

        if self.block:
            self.downsample = (
                Seq()
                .append(
                    self.CONVOLUTION(
                        in_channels=down_conv_nn[0],
                        out_channels=down_conv_nn[1],
                        kernel_size=3,
                        stride=stride,
                        dilation=dilation,
                        has_bias=False,
                        dimension=dimension,
                    )
                )
                .append(ME.MinkowskiBatchNorm(down_conv_nn[1]))
            )
        else:
            self.downsample = None

    def forward(self, x):
        out = self.conv_in(x)
        if self.block:
            residual = self.downsample(x)
            out = self.block(out) + residual
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
        out = self.conv_in(inp)
        if self.block:
            residual = self.downsample(inp)
            out = self.block(out) + residual
        return out
