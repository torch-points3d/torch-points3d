from abc import ABC, abstractmethod
import math
from functools import partial
import torch
from torch_geometric.nn import fps, radius, knn


class BaseSampler(ABC):

    def __init__(self, ratio=None, num_to_sample=None):
        '''If num_to_sample is provided, sample exactly 
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
        '''
        self.ratio = ratio
        self.num_to_sample = num_to_sample

    def __call__(self, pos, batch):
        return self.sample(pos, batch)

    def get_num_to_sample(self, pos):
        if self.num_to_sample is not None:
            return self.num_to_sample
        else:
            return math.floor(pos.shape[0]*self.ratio)

    @abstractmethod
    def sample(self, pos, batch):
        pass


class FPSSampler(BaseSampler):

    def __init__(self, ratio):
        super(FPSSampler, self).__init__(ratio)

    def sample(self, pos, batch):
        return fps(pos, batch, ratio=self.ratio)


class RandomSampler(BaseSampler):

    def __init__(self, ratio=None, num_to_sample=None):
        super(RandomSampler, self).__init__(ratio, num_to_sample)

    def sample(self, pos, batch):
        idx = torch.randint(0, pos.shape[0], self.get_num_to_sample(pos))
        return idx


class BaseNeighbourFinder(ABC):

    def __call__(self, x, y, batch_x, batch_y):
        return self.find_neighbours(x, y, batch_x, batch_y)

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

class DilatedKNNNeighbourFinder(BaseNeighbourFinder):

    def __init__(self, k, dilation):
        self.k = k 
        self.dilation = dilation
        self.initialFinder = KNNNeighbourFinder(k * dilation)

    def find_neighbours(self, x, y, batch_x, batch_y):
        #find the self.k * self.dilation closest neighbours in x for each y
        row, col = self.initialFinder.find_neighbours(x, y, batch_x, batch_y) 

        #for each point in y, randomly select k of its neighbours
        index = torch.randint(self.k * self.dilation, (len(y), self.k), device = row.device, dtype = torch.long)

        arange = torch.arange(len(y), dtype=torch.long, device = row.device)
        arange = arange * (self.k * self.dilation) 
        index = (index + arange.view(-1, 1)).view(-1)
        row, col = row[index], col[index]

        return row, col
