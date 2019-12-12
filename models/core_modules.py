from abc import ABC, abstractmethod
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LeakyReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import knn_interpolate, fps, radius, global_max_pool, global_mean_pool


def MLP(channels, activation=ReLU()):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), activation, BN(channels[i]))
        for i in range(1, len(channels))
    ])


class FPModule(torch.nn.Module):
    """ Upsampling module from PointNet++

    Arguments:
        k [int] -- number of nearest neighboors used for the interpolation
        up_conv_nn [List[int]] -- list of feature sizes for the uplconv mlp

    Returns:
        [type] -- [description]
    """

    def __init__(self, up_k, up_conv_nn, *args, **kwargs):
        super(FPModule, self).__init__()
        self.k = up_k
        self.nn = MLP(up_conv_nn)

    def forward(self, data):
        # print([x.shape if x is not None else x for x in data])
        x, pos, batch, x_skip, pos_skip, batch_skip = data
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        data = (x, pos_skip, batch_skip)
        return data


class BaseConvolution(ABC, torch.nn.Module):
    def __init__(self, ratio, radius, *args, **kwargs):
        torch.nn.Module.__init__(self)

        self.ratio = ratio
        self.radius = radius
        self.max_num_neighbors = kwargs.get("max_num_neighbors", 64)

    @property
    @abstractmethod
    def conv(self):
        pass

    def forward(self, data, returnIdx=False):
        x, pos, batch = data
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.radius, batch, batch[idx],
                          max_num_neighbors=self.max_num_neighbors)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        data = (x, pos, batch)

        if returnIdx:
            return data, idx
        return data

class BaseIdentityConvolution(ABC, torch.nn.Module):
    '''
        Base class for convolutions that do not downsample the point cloud.
    '''
    
    def __init__(self, radius, *args, **kwargs):
        super(BaseIdentityConvolution, self).__init__()

        self.radius = radius
        self.max_num_neighbors = kwargs.get('max_num_neighbours', 64)

    @property
    @abstractmethod
    def conv(self):
        pass

    def forward(self, x, pos, batch):
        row, col = radius(pos, pos, self.radius, batch, batch, self.max_num_neighbors)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos), edge_index)
        return x

class BaseResnetBlock(ABC, torch.nn.Module):

    def __init__(self, indim, convdim, outdim, *args, **kwargs):
        super(BaseResnetBlock, self).__init__()

        self.indim = indim
        self.convdim = convdim
        self.outdim = outdim

        self.features_downsample_nn = MLP([self.indim, self.indim//2])
        self.features_upsample_nn = MLP([self.indim//2, self.outdim])

        self.shortcut_feature_resize_nn = MLP([self.indim, self.outdim])

        self.activation = ReLU()


class BaseIdentityResnet(torch.nn.Module):
    
    def __init__(self, conv1, conv2, dim, *args, **kwargs):
        super(BaseIdentityResnet, self).__init__()

        self.conv1 = conv1
        self.conv2 = conv2
        self.dim = dim

        self.features_downsample_nn = MLP([self.dim, self.dim//2])
        self.features_upsample_nn = MLP([self.dim//2, self.dim])

        self.shortcut_feature_resize_nn = MLP([self.])

        self.activation = ReLU()

    def forward(self, data):

        x, pos, batch = data

        shortcut = x

        x = self.features_downsample_nn(x)

        x = self.conv1(x, pos, batch)

        x = self.conv2(x, pos, batch)

        x = shortcut + x

        return self.activation(x)

class BaseConvResnet(torch.nn.Module):

    def __init__(self, conv1, indim, outdim, *args, **kwargs):
        super(BaseConvResnet, self).__init__()

        self.conv1 = conv1
        self.indim = indim
        self.outdim = outdim

        self.features_dsample_nn = MLP([self.indim, self.indim//2])
        self.features_upsample_nn = MLP([self.indim//2, self.outdim])

        self.shortcut_feature_resize_nn = MLP([self.indim, self.outdim])

        self.activation = ReLU()

    def forward(self, data):

        x, pos, batch = data

        shortcut = x

        x = self.features_dsample_nn(x)

        (x, pos, batch), idx = self.conv1((x, pos, batch), returnIdx=True)

        x = self.features_upsample_nn(x)

        shortcut = self.shortcut_feature_resize_nn(shortcut[idx])

        x = shortcut[idx] + x

        return self.activation(x)


class GlobalBaseModule(torch.nn.Module):
    def __init__(self, nn, aggr='max'):
        super(GlobalBaseModule, self).__init__()
        self.nn = MLP(nn)
        self.pool = global_max_pool if aggr == "max" else global_mean_pool

    def forward(self, data):
        x, pos, batch = data
        x = self.nn(torch.cat([x, pos], dim=1))
        x = self.pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        data = (x, pos, batch)
        return data
