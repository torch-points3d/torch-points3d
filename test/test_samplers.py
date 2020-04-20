import os
import sys
import unittest
import numpy as np


ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ROOT)

from torch_points3d.datasets.samplers import BalancedRandomSampler


class TestBalancedRandomSampler(unittest.TestCase):
    def test_simple(self):

        num_classes = 10
        num_samples = 10000

        p = np.asarray([2 ** i for i in range(num_classes)]).astype(float)
        p /= p.sum()

        labels = np.random.choice(range(num_classes), num_samples, p=p)

        sampler = BalancedRandomSampler(labels)

        indices = iter(sampler)
        _, c = np.unique(labels[list(indices)], return_counts=True)
        self.assertGreater(0.005, np.std(c) / num_samples)


if __name__ == "__main__":
    unittest.main()
