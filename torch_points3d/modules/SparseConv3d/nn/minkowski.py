import torch
import MinkowskiEngine as ME


class Conv3d(ME.MinkowskiConvolution):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            has_bias=bias,
            dimension=3,
        )


class Conv3dTranspose(ME.MinkowskiConvolutionTranspose):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            has_bias=bias,
            dimension=3,
        )


class BatchNorm(ME.MinkowskiBatchNorm):
    pass


class ReLU(ME.MinkowskiReLU):
    def __init__(self, inplace=False):
        super().__init__(inplace=False)


def cat(*args):
    return ME.cat(*args)


def SparseTensor(feats, coordinates, device=torch.device("cpu")):
    return ME.SparseTensor(feats, coords=coordinates).to(device)
