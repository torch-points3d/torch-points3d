import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import (
    Sequential as Seq,
    Linear as Lin,
    ReLU,
    LeakyReLU,
    BatchNorm1d as BN,
    Dropout,
)


def weight_variable(shape):

    initial = torch.empty(shape, dtype=torch.float)
    torch.nn.init.xavier_normal_(initial)
    return initial


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, data):
        return data


def MLP(channels, activation=nn.LeakyReLU(0.2), bn_momentum=0.1, bias=True):
    return Seq(
        *[
            Seq(Lin(channels[i - 1], channels[i], bias=bias), activation, BN(channels[i], momentum=bn_momentum))
            for i in range(1, len(channels))
        ]
    )


class UnaryConv(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        """
        1x1 convolution on point cloud (we can even call it a mini pointnet)
        """
        super(UnaryConv, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.weight = Parameter(weight_variable([self.num_inputs, self.num_outputs]))

    def forward(self, features):
        """
        features(Torch Tensor): size N x d d is the size of inputs
        """
        return torch.matmul(features, self.weight)

    def __repr__(self):
        return "UnaryConv({}, {})".format(self.num_inputs, self.num_outputs)
