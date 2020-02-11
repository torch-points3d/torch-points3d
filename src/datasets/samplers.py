import torch
import numpy as np
from torch.utils.data import Sampler

class BalancedRandomSampler(Sampler):
    r"""This sampler is responsible for creating balanced batch based on the class distribution.
    It is implementing a replacement=True strategy for indices selection
    """
    def __init__(self, labels, replacement=True):

        self.num_samples = len(labels)

        self.idx_classes, self.counts = np.unique(labels, return_counts=True)
        self.indices = {
           idx: np.argwhere(labels == idx).flatten() for idx in self.idx_classes
        }

    def __iter__(self):
        indices = []
        for _ in range(self.num_samples):
            idx = np.random.choice(self.idx_classes)
            indice = int(np.random.choice(self.indices[idx]))
            indices.append(indice)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def __repr__(self):
        return "{}(num_samples={})".format(self.__class__.__name__, self.num_samples)

