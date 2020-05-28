import unittest
import torch
from torch_geometric.data import Data

from torch_points3d.core.spatial_ops import KNNInterpolate
from torch_points3d.core.data_transform import GridSampling3D


class TestInterpolate(unittest.TestCase):
    def test_precompute(self):
        pos = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0], [0.1, 0, 0]])
        x = torch.tensor([0, 0, 0, 0, 1]).unsqueeze(-1)
        support = Data(x=x, pos=pos)

        query = GridSampling3D(1)(support.clone())

        interpolate = KNNInterpolate(1)
        up = interpolate.precompute(query, support)
        self.assertEqual(up.num_nodes, 5)
        self.assertEqual(up.x_idx[4], up.x_idx[3])

    def test_compute(self):
        npoints = 100
        pos = torch.randn((npoints, 3))
        x = torch.randn((npoints, 4))
        support = Data(x=x, pos=pos)
        query = Data(x=torch.randn((npoints // 2, 4)), pos=torch.randn((npoints // 2, 3)))

        interpolate = KNNInterpolate(3)
        precomputed = interpolate.precompute(query, support)

        gt = interpolate(query, support)
        pre = interpolate(query, support, precomputed=precomputed)

        torch.testing.assert_allclose(gt, pre)
