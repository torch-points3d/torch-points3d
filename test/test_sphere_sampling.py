import os
import sys
import unittest
import numpy as np
import torch
from torch_geometric.data import Data

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
from torch_points3d.core.data_transform.transforms import RandomSphere, SphereSampling


class TestRandomSphere(unittest.TestCase):
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


class TestSphereSampling(unittest.TestCase):
    def setUp(self):
        pos = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1]])
        labels = torch.tensor([0, 1, 2, 0, 0, 0])
        self.data = Data(pos=pos.float(), labels=labels)

    def test_sphere(self):
        sphere_sampling = SphereSampling(0.1, [0, 0, 0])
        sampled = sphere_sampling(self.data)

        self.assertIn(SphereSampling.KDTREE_KEY, self.data)
        self.assertEqual(len(sampled.labels), 1)
        self.assertEqual(sampled.labels[0], 2)

    def test_align(self):
        sphere_sampling = SphereSampling(0.1, [1, 0, 0])
        sampled = sphere_sampling(self.data)
        torch.testing.assert_allclose(sampled.pos, torch.tensor([[0.0, 0, 0]]))
        self.assertEqual(sampled.labels[0], 0)

        sphere_sampling = SphereSampling(0.1, [1, 0, 0], align_origin=False)
        sampled = sphere_sampling(self.data)
        torch.testing.assert_allclose(sampled.pos, torch.tensor([[1.0, 0, 0]]))
        self.assertEqual(sampled.labels[0], 0)


if __name__ == "__main__":
    unittest.main()
