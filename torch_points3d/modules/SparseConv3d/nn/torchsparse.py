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
            transposed=True,
        )


class BatchNorm(torch.nn.Module):
    def __init__(self, num_features: int, *, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.bn = TS.nn.BatchNorm(num_features=num_features, eps=eps, momentum=momentum)

    def forward(self, feats):
        return self.bn(feats)

    def __repr__(self):
        return self.bn.__repr__()


class ReLU(TS.nn.ReLU):
    def __init__(self, inplace=True):
        super().__init__(inplace=inplace)


def cat(*args):
    return TS.cat(args)


def SparseTensor(feats, coordinates, batch, device=torch.device("cpu")):
    if batch.dim() == 1:
        batch = batch.unsqueeze(-1)
    coords = torch.cat([coordinates.int(), batch.int()], -1)
    return TS.SparseTensor(feats, coords).to(device)
