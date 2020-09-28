import copy
from abc import abstractmethod
from collections import deque

import torch.nn as nn

from torch_points3d.modules.SPVCNN.modules.modules import RandomModule

__all__ = ['RandomNet']


class RandomNet(nn.Module):
    def random_sample(self):
        sample = {}
        for name, module in self.named_random_modules():
            sample[name] = module.random_sample()
        return sample

    def manual_select(self, sample):
        for name, module in self.named_random_modules():
            module.manual_select(sample[name])

    def named_random_modules(self):
        for name, module in self.named_modules():
            if isinstance(module, RandomModule):
                yield name, module

    def random_modules(self):
        for name, module in self.named_random_modules():
            yield module

    @abstractmethod
    def forward(self, *inputs):
        pass

    def determinize(self):
        model = copy.deepcopy(self)
        queue = deque([model])
        while queue:
            x = queue.popleft()
            for name, module in x._modules.items():
                while isinstance(module, RandomModule):
                    module = x._modules[name] = module.determinize()
                queue.append(module)
        return model
