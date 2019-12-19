from abc import ABC, abstractmethod
import math
from functools import partial
import torch
from torch_geometric.nn import fps, radius, knn


class BaseSampler(ABC):

    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, pos, batch):
        return self.sample(pos, batch)

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
