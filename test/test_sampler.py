import os
import unittest
import torch

from models.core_sampling_and_search import (
    FPSSampler,
    RandomSampler,
    RadiusNeighbourFinder,
    MultiscaleRadiusNeighbourFinder,
)


class TestSampler(unittest.TestCase):
    def test_fps(self):
        pos = torch.randn((10, 3))
        batch = torch.zeros(10).to(torch.long)
        sampler = FPSSampler(ratio=0.5)
        self.assertEqual(sampler(pos, batch).shape[0], 5)

        sampler = FPSSampler(num_to_sample=5)
        self.assertEqual(sampler(pos, batch).shape[0], 5)

    def test_random(self):
        pos = torch.randn((10, 3))
        batch = torch.zeros(10).to(torch.long)
        sampler = RandomSampler(ratio=0.5)
        self.assertEqual(sampler(pos, batch).shape[0], 5)

        sampler = RandomSampler(num_to_sample=5)
        self.assertEqual(sampler(pos, batch).shape[0], 5)


class TestNeighboorhoodSearch(unittest.TestCase):
    def test_single_search(self):
        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.Tensor([[-1, 0], [1, 0]])
        batch_y = torch.tensor([0, 0])

        nei_finder = MultiscaleRadiusNeighbourFinder(1, 4)
        self.assertEqual(nei_finder(x, y, batch_x, batch_y, 0)[1, :].shape[0], 4)

    def test_multi_radius_search(self):
        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.Tensor([[-1, 0], [1, 0]])
        batch_y = torch.tensor([0, 0])
        nei_finder = MultiscaleRadiusNeighbourFinder([1, 10], 4)
        multiscale = []
        for i in range(2):
            multiscale.append(nei_finder(x, y, batch_x, batch_y, i))

        self.assertEqual(len(multiscale), 2)
        self.assertEqual(multiscale[0][1, :].shape[0], 4)
        self.assertEqual(multiscale[1][1, :].shape[0], 8)

    def test_multi_num_search(self):
        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.Tensor([[-1, 0], [1, 0]])
        batch_y = torch.tensor([0, 0])
        nei_finder = MultiscaleRadiusNeighbourFinder(10, [3, 4])
        multiscale = []
        for i in range(2):
            multiscale.append(nei_finder(x, y, batch_x, batch_y, i))

        self.assertEqual(len(multiscale), 2)
        self.assertEqual(multiscale[0][1, :].shape[0], 6)
        self.assertEqual(multiscale[1][1, :].shape[0], 8)

    def test_multiall(self):
        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.Tensor([[-1, 0], [1, 0]])
        batch_y = torch.tensor([0, 0])

        nei_finder = MultiscaleRadiusNeighbourFinder([1, 10], [3, 4])
        multiscale = []
        for i in range(2):
            multiscale.append(nei_finder(x, y, batch_x, batch_y, i))

        self.assertEqual(len(multiscale), 2)
        self.assertEqual(multiscale[0][1, :].shape[0], 4)
        self.assertEqual(multiscale[1][1, :].shape[0], 8)

    def test_raises(self):
        with self.assertRaises(ValueError):
            nei_finder = MultiscaleRadiusNeighbourFinder([1], [3, 4])

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.Tensor([[-1, 0], [1, 0]])
        batch_y = torch.tensor([0, 0])
        nei_finder = MultiscaleRadiusNeighbourFinder([1, 2], [3, 4])
        with self.assertRaises(ValueError):
            nei_finder(x, y, batch_x, batch_y, 10)


if __name__ == "__main__":
    unittest.main()
