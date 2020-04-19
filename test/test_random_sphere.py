import os
import sys
import unittest
import numpy as np
import torch
from torch_geometric.data import Data

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points3d.core.data_transform.transforms import RandomSphere


class TestRandmSphere(unittest.TestCase):
    def setUp(self):

        pos = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1]])
        labels = np.array([0, 0, 0, 0, 0, 0])

        self.data = Data(pos=torch.from_numpy(pos).float(), labels=torch.from_numpy(labels))

    def test_neighbour_found_under_random_sampling(self):
        random_sphere = RandomSphere(0.1, strategy="RANDOM")
        data = random_sphere(self.data.clone())
        assert data.labels.shape[0] == 1

        random_sphere = RandomSphere(3, strategy="RANDOM")
        data = random_sphere(self.data.clone())
        assert data.labels.shape[0] == 6


if __name__ == "__main__":
    unittest.main()
