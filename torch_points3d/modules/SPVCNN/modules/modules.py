import random
from abc import abstractmethod

import torch.nn as nn

__all__ = ['RandomModule', 'RandomChoice', 'RandomDepth']


class RandomModule(nn.Module):
    @abstractmethod
    def random_sample(self):
        pass

    @abstractmethod
    def clear_sample(self):
        pass

    @abstractmethod
    def manual_select(self, sample):
        pass

    def forward(self, *inputs):
        return self.determinize()(*inputs)

    @abstractmethod
    def determinize(self):
        pass


class RandomChoice(RandomModule):
    def __init__(self, *choices):
        super().__init__()
        self.choices = nn.ModuleList(choices)

    def random_sample(self):
        self.index = random.randint(0, len(self.choices) - 1)
        return self.index

    def clear_sample(self):
        self.index = None

    def manual_select(self, index):
        self.index = index

    def determinize(self):
        return self.choices[self.index]


class RandomDepth(RandomModule):
    def __init__(self, *layers, depth_min=None, depth_max=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.depth_min = depth_min
        self.depth_max = depth_max

    def random_sample(self):
        if self.depth_min is not None:
            depth_min = self.depth_min
        else:
            depth_min = 0

        if self.depth_max is not None:
            depth_max = self.depth_max
        else:
            depth_max = len(self.layers)

        self.depth = random.randint(depth_min, depth_max)
        return self.depth

    def clear_sample(self):
        self.depth = None

    def status(self):
        return self.depth

    def manual_select(self, depth):
        self.depth = depth

    # fixme: support tuples as input
    def forward(self, x):
        for k in range(self.depth):
            x = self.layers[k](x)
        return x

    def determinize(self):
        return nn.Sequential(*self.layers[:self.depth])
