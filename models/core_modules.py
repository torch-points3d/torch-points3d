from abc import ABC, abstractmethod
import math
from functools import partial
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LeakyReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import knn_interpolate, fps, radius, global_max_pool, global_mean_pool, knn


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
        print([x.shape if x is not None else x for x in data])
        x, pos, batch, x_skip, pos_skip, batch_skip = data
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        data = (x, pos_skip, batch_skip)
        return data
        
class BaseConvolution(ABC, torch.nn.Module):
    def __init__(self, sampler, neighbour_finder, *args, **kwargs):
        torch.nn.Module.__init__(self)

        self.sampler = sampler
        self.neighbour_finder = neighbour_finder

    @property
    @abstractmethod
    def conv(self):
        pass

    def forward(self, data):
        x, pos, batch = data
        idx = self.sampler(pos, batch)
        row, col = self.neighbour_finder(pos, pos[idx], batch, batch[idx])
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        data = (x, pos, batch, idx)
        return data

# class BaseKNNConvolution(ABC, torch.nn.Module):

#     def __init__(self, ratio=None, k=None, sampling_strategy = None, *args, **kwargs):
#         torch.nn.Module.__init__(self)

#         self.ratio = ratio
#         self.k = k
#         self.sampling_strategy = sampling_strategy

#     @abstractmethod
#     def conv(self, x, pos, edge_index):
#         pass 

#     def forward(self, data):
#         x, pos, batch = data

#         if self.ratio == 1: #convolve every point
#             row, col = knn(pos, pos, self.k, batch, batch)
#             edge_index = torch.stack([col, row], dim=0)
#             x = self.conv(x, (pos, pos), edge_index)
#             return x, pos, batch, None
#         else: #downsample using self.sampling_strategy and convolve 
#             if self.sampling_strategy == 'fps':
#                 idx = fps(pos, batch, self.ratio)
#             elif self.sampling_strategy == 'random':
#                 idx = torch.randint(0, pos.shape[0], (math.floor(pos.shape[0]*self.ratio),))
#             else:
#                 raise ValueError("Unrecognised sampling_strategy: " + self.sampling_strategy)
            
#             row, col = knn(pos, pos[idx], self.k, batch, batch[idx])
#             edge_index = torch.stack([col, row], dim=0)
#             x = self.conv(x, (pos, pos[idx]), edge_index)
#             pos, batch = pos[idx], batch[idx]
#             return x, pos, batch, idx


class BaseResnetBlock(ABC, torch.nn.Module):

    def __init__(self, indim, outdim, convdim):
        '''
            indim: size of x at the input
            outdim: desired size of x at the output
            convdim: size of x following convolution
        '''
        torch.nn.Module.__init__(self)

        self.indim = indim
        self.outdim = outdim
        self.convdim = convdim

        self.features_downsample_nn = MLP([self.indim, self.outdim//4])
        self.features_upsample_nn = MLP([self.convdim, self.outdim])

        self.shortcut_feature_resize_nn = MLP([self.indim, self.outdim])

        self.activation = ReLU()

    @abstractmethod
    def convolution(self, data):
        pass

    def forward(self, data):

        x, pos, batch = data #(N, indim)

        shortcut = x #(N, indim)

        x = self.features_downsample_nn(x) #(N, outdim//4)

        #if this is an identity resnet block, idx will be None
        x, pos, batch, idx = self.convolution((x, pos, batch)) #(N', convdim)

        x = self.features_upsample_nn(x) #(N', outdim)

        if idx is not None:
            shortcut = shortcut[idx] #(N', indim)

        shortcut = self.shortcut_feature_resize_nn(shortcut) #(N', outdim)

        x = shortcut + x

        return self.activation(x), pos, batch


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



class BaseSampler(ABC):

    def __init__(self, ratio):
        self.ratio = ratio

    @abstractmethod
    def sample(self, pos, batch):
        pass

class FPSSampler(BaseSampler):

    def __init__(self, ratio):
        super(FPSSampler, self).__init__(ratio)

    def sample(self, pos, batch):
        return fps(pos, batch, ratio=self.ratio)

class RandomSampler(BaseSampler):

    def __init__(self, ratio):
        super(RandomSampler, self).__init__(ratio)

    def sample(self, pos, batch):
        idx = torch.randint(0, pos.shape[0], (math.floor(pos.shape[0]*self.ratio),))
        return idx



class BaseNeighbourFinder(ABC):

    @abstractmethod
    def find_neighbours(self, x, y, batch_x, batch_y):
        pass

class RadiusNeighbourFinder(BaseNeighbourFinder):

    def __init__(self, radius, max_num_neighbors=64):
        self.radius = radius 
        self.max_num_neighbors = max_num_neighbors

    def find_neighbours(self, x, y, batch_x, batch_y):  
        return radius(x, y, self.radius, batch_x, batch_y, max_num_neighbors=self.max_num_neighbors)

class KNNNeighbourFinder(BaseNeighbourFinder):

    def __init__(self, k):
        self.k = k 

    def find_neighbours(self, x, y, batch_x, batch_y):
        return knn(x, y, self.k, batch_x, batch_y)      

# class Sampler():

#     def __init__(self, sampling_strategy = 'fps', neighbour_strategy = 'radius', ratio = None, opts = None):

#         if sampling_strategy == 'fps':
#             self.sampling_func = partial(fps, ratio=ratio)

#         elif sampling_strategy == 'random':

#             def random_sample(pos, batch):
#                 idx = torch.randint(0, pos.shape[0], (math.floor(pos.shape[0]*ratio),))
#                 return idx

#             self.sampling_func = random_sample
#         else:
#             raise ValueError("Unrecognised sampling strategy: ", sampling_strategy)

#         if neighbour_strategy == 'radius':

#             def radius_wrapper(x, y, batch_x, batch_y):
#                 return radius(x, y, opts['radius'], batch_x, batch_y, max_num_neighbors=opts['max_num_neighbours'])

#             self.neighbour_func = radius_wrapper
#         elif neighbour_strategy == 'knn':

#             def knn_wrapper(x, y, batch_x, batch_y):
#                 return knn(x, y, opts['k'], batch_x, batch_y)

#             self.neighbour_func = knn_wrapper
#         else:
#             raise ValueError("Unrecognised neighbour strategy: ", neighbour_strategy)

#     def build(self):

#         def sample_func(pos, batch):
#             idx = self.sampling_func(pos, batch)
#             row, col = self.neighbour_func(pos, pos[idx], batch, batch[idx])
#             edge_index = torch.stack([col, row], dim=0)
#             return edge_index

#         return sample_func
            


