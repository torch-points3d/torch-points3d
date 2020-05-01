import torch.nn as nn
from .base_modules import Seq


class Conv2D(Seq):
    def __init__(self, in_channels, out_channels, bias=True, bn=True, activation=nn.LeakyReLU(negative_slope=0.01)):
        super().__init__()
        self.append(nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=bias))
        if bn:
            self.append(nn.BatchNorm2d(out_channels))
        if activation:
            self.append(activation)


class Conv1D(Seq):
    def __init__(self, in_channels, out_channels, bias=True, bn=True, activation=nn.LeakyReLU(negative_slope=0.01)):
        super().__init__()
        self.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias))
        if bn:
            self.append(nn.BatchNorm1d(out_channels))
        if activation:
            self.append(activation)


class MLP2D(Seq):
    def __init__(self, channels, bias=False, bn=True, activation=nn.LeakyReLU(negative_slope=0.01)):
        super().__init__()
        for i in range(len(channels) - 1):
            self.append(Conv2D(channels[i], channels[i + 1], bn=bn, bias=bias, activation=activation))
