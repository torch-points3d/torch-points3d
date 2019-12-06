import torch
import torch_geometric
from torch_geometric.nn import global_max_pool, global_mean_pool, fps, radius, knn_interpolate
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LeakyReLU, BatchNorm1d as BN

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), LeakyReLU(0.2))
        for i in range(1, len(channels))
    ])

class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip

class BaseConvolution(torch.nn.Module):
    def __init__(self, ratio, radius):
        super(BaseConvolution, self).__init__()
        self.ratio = ratio
        self.radius = radius
        self.conv = None # This one should be implemented

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.radius, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class GlobalBaseModule(torch.nn.Module):
    def __init__(self, nn, aggr='max'):
        super(GlobalBaseModule, self).__init__()
        self.nn = nn
        self.pool = global_max_pool if aggr == "max" else  global_mean_pool

    def forward(self, x, pos, batch):
        print(x.shape, pos.shape)
        x = self.nn(torch.cat([x, pos], dim=1))
        x = self.pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch