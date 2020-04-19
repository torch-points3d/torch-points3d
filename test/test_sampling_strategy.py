import os
import sys
import unittest
import numpy as np
import torch
from torch_geometric.data import Data

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points3d.utils.transform_utils import SamplingStrategy


class TestSamplingStrategy(unittest.TestCase):
    def setUp(self):

        pos = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1]])
        self.labels = np.array([0, 1, 2, 3, 4, 5])

        self.data = Data(pos=torch.from_numpy(pos).float(), labels=torch.from_numpy(self.labels))

    def test_random_sampling_strategy(self):
        random_sphere = SamplingStrategy(strategy="RANDOM")

        np.random.seed(42)

        random_labels = []
        for i in range(50):
            random_center = random_sphere(self.data.clone())
            random_labels.append(self.labels[random_center])

        assert len(np.unique(random_labels)) == len(self.labels)


if __name__ == "__main__":
    unittest.main()
