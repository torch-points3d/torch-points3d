import unittest
import torch
from torch_geometric.data import Data


from torch_points3d.core.spatial_ops import (
    FPSSampler,
    RandomSampler,
    RadiusNeighbourFinder,
    MultiscaleRadiusNeighbourFinder,
)
from torch_points3d.modules.VoteNet.dense_samplers import FPSSamplerToDense


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

        nei_finder = MultiscaleRadiusNeighbourFinder(1.1, 4)
        self.assertEqual(nei_finder(x, y, batch_x, batch_y, 0)[1, :].shape[0], 4)

    def test_multi_radius_search(self):
        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.Tensor([[-1, 0], [1, 0]])
        batch_y = torch.tensor([0, 0])
        nei_finder = MultiscaleRadiusNeighbourFinder([1.1, 10], 4)
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

        nei_finder = MultiscaleRadiusNeighbourFinder([1.1, 10], [3, 4])
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


class TestFPStoDense(unittest.TestCase):
    def test_dense(self):
        data = Data(pos=torch.randn((3, 1000, 3)), x=torch.randn((3, 3, 1000)))
        sampler = FPSSamplerToDense(num_to_sample=201)
        sampled_data, idx = sampler.sample(data, 3, "DENSE")
        self.assertEqual(sampled_data.pos.shape, (3, 201, 3))
        self.assertEqual(sampled_data.x.shape, (3, 3, 201))

    def test_packed(self):
        batch = torch.cat([0 * torch.ones(200), torch.ones(801), 2 * torch.ones(1999)])
        data = Data(pos=torch.randn((3000, 3)), x=torch.randn((3000, 10)), batch=batch)
        sampler = FPSSamplerToDense(num_to_sample=150)
        sampled_data, idx = sampler.sample(data, 3, "PARTIAL_DENSE")
        self.assertEqual(sampled_data.pos.shape, (3, 150, 3))
        self.assertEqual(sampled_data.x.shape, (3, 10, 150))


if __name__ == "__main__":
    unittest.main()
