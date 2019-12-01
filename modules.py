import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import math
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, MessagePassing
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

        # Init the kernel using attrative + repulsion forces
        kernel, _ = kernel_point_optimization_debug(self.radius, self.num_points, num_kernels=1, \
            dimension=self.kernel_dim, fixed=self.fixed, ratio=self.ratio, verbose=self.verbose)
        self.kernel.data = torch.from_numpy(kernel)

class PointsKernel(MessagePassing):

    def __init__(self, num_kernels, num_points, in_features, out_features, KP_influence='linear', aggregation_mode='closest', KP_extent=0.2, \
        bias=True, kernel_dim=3, radius=1., fixed='center', ratio=1.0, verbose=0, aggr='mean', **kwargs):
        super(PointsKernel, self).__init__(aggr=aggr, **kwargs)
        self.num_kernels = num_kernels
        self.num_points = num_points
        self.in_features = in_features
        self.out_features = out_features
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.KP_extent = KP_extent
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

    def forward(self, x, pos, edge_index, batch):
        self.x_is_none = x is None
        if not self.x_is_none:
            return self.propagate(edge_index, x=torch.cat([x, pos]))
        else:
            return self.propagate(edge_index, x=pos.float())

    def get_point_kernels(self):
        return torch.cat([pk.kernel for pk in self.kernels])

    def get_point_kernels_weight(self):
        return torch.cat([pk.kernel_weight for pk in self.kernels])

    def message(self, x_i, x_j):
        if self.x_is_none:
            neighborhood_features = x_j
        else:
            neighborhood_features = x_j[:, :-self.kernel_dim]
            x_i, x_j = x_i[:, -self.kernel_dim:], x_j[:, -self.kernel_dim:]

        # Center every neighborhood
        neighbors = x_j -  x_i
        
        #Get points kernels
        K_points = self.get_point_kernels()

        # Get all difference matrices [n_neighbors, n_kpoints, dim]
        neighbors = neighbors.unsqueeze(1)

        differences = neighbors - K_points.float().view((-1, 3)).unsqueeze(0)
        sq_distances = F.normalize(differences, p=2, dim=-1).sum(-1)

        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)

        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = 1 - torch.sqrt(sq_distances) / self.KP_extent
            all_weights[all_weights < 0] = 0
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')
        
        closest_kernels_idx = torch.argmin(sq_distances, dim=-1)
        all_weights =  torch.gather(all_weights, -1, closest_kernels_idx.unsqueeze(-1))
        
        K_weights = self.get_point_kernels_weight()
        K_weights = torch.index_select(K_weights, 0, closest_kernels_idx)

        # Apply distance weights [n_points, n_kpoints, in_fdim]
        weighted_neighborhood_features = all_weights * neighborhood_features

        # Apply network weights [n_kpoints, n_points, out_fdim]
        print(K_weights.shape, weighted_neighborhood_features.shape)
        out = torch.einsum('nab,nb->na', K_weights, weighted_neighborhood_features)
        print(out.shape)
        return out

    """
        features = tf.concat([features, tf.zeros_like(features[:1, :])], axis=0)

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        neighborhood_features = tf.gather(features, neighbors_indices, axis=0)

        # Apply distance weights [n_points, n_kpoints, in_fdim]
        weighted_features = tf.matmul(all_weights, neighborhood_features)

        # Apply network weights [n_kpoints, n_points, out_fdim]
        weighted_features = tf.transpose(weighted_features, [1, 0, 2])
        kernel_outputs = tf.matmul(weighted_features, K_values)

        # Convolution sum to get [n_points, out_fdim]
        output_features = tf.reduce_sum(kernel_outputs, axis=0)

        return output_features
    """

    def update(self, aggr_out):
        return aggr_out

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

        self.points_kernel = PointsKernel(self.num_kernels, self.num_points, self.in_features, self.out_features, bias=self.bias, kernel_dim=self.kernel_dim, \
            radius=self.radius, fixed=self.fixed, ratio=self.ratio, verbose=self.verbose)

    def forward(self, x, pos, batch):
        row, col = radius(pos, pos, self.radius, batch, batch,
                          max_num_neighbors=self.num_points)        
        edge_index = torch.stack([col, row], dim=0)

        self.points_kernel(x, pos, edge_index, batch)        

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
