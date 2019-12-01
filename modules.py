import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import math
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch.nn.parameter import Parameter

from kernel_utils import kernel_point_optimization_debug

class PointKernel(nn.Module):

    def __init__(self, in_features, out_features, num_points, bias=True, kernel_dim=3, radius=1., fixed='center', ratio=1.0, verbose=0):
        super(PointKernel, self).__init__()
        
        # PointKernel parameters
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.num_points = num_points
        self.kernel_dim = kernel_dim
        self.radius = radius
        self.fixed = fixed
        self.ratio = ratio
        self.verbose = verbose

        # Point position in kernel_dim
        self.kernel = Parameter(torch.Tensor(1, num_points, kernel_dim))
        
        # Associated weights
        self.kernel_weight = Parameter(torch.Tensor(num_points, out_features, in_features))
        if bias:
            self.kernel_bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('kernel_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.kernel_weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.kernel_weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.kernel_bias, -bound, bound)

        kernel, _ = kernel_point_optimization_debug(self.radius, self.num_points, num_kernels=1, \
            dimension=self.kernel_dim, fixed=self.fixed, ratio=self.ratio, verbose=self.verbose)

        print(kernel.shape)
        self.kernel.data = torch.from_numpy(kernel)


class KPConv(nn.Module):

    def __init__(self, num_kernels, num_points, in_features, out_features, bias=True, kernel_dim=3, radius=1., fixed='center', ratio=1.0, verbose=0):
        super(KPConv, self).__init__()
        self.num_kernels = num_kernels
        self.num_points = num_points
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.kernel_dim = kernel_dim
        self.radius = radius
        self.fixed = fixed
        self.ratio = ratio
        self.verbose = verbose

        self.kernels = nn.ModuleList() 
        for _ in range(self.num_kernels):
            self.kernels.append(PointKernel(self.in_features, self.out_features, self.num_points, bias=self.bias, kernel_dim=self.kernel_dim, \
            radius=self.radius, fixed=self.fixed, ratio=self.ratio, verbose=self.verbose))

    def forward(self, x, pos, batch):
        row, col = radius(pos, pos, self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

    






class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])
