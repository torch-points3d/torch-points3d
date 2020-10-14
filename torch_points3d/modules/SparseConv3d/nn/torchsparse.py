import torch
import torchsparse as TS
import torchsparse.nn


class Conv3d(TS.nn.Conv3d):
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
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias,
        )


class Conv3dTranspose(TS.nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        transpose: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            transpose=True,
        )


class BatchNorm(TS.nn.BatchNorm):
    pass


class ReLU(TS.nn.ReLU):
    pass


def cat(*args, dim=1):
    return TS.cat(args, dim)


def SparseTensor(feats, coordinates, device=torch.device("cpu")):
    return TS.SparseTensor(feats, coordinates).to(device)
