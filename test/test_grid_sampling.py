import os
import sys
import numpy as np
import torch
import unittest
import h5py
import numpy.testing as npt
import numpy.matlib
from torch_geometric.data import Data

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, ".."))

import torch_points3d.core.data_transform as cT


class TestGridSampling3D(unittest.TestCase):
    def setUp(self):
        self.sampler = cT.GridSampling3D(0.04)
        num_points = 5
        pos = torch.from_numpy(np.array([[0, 0, 0.01], [0.01, 0, 0], [0, 0.01, 0], [0, 0.01, 0], [0.01, 0, 0.01]]))
        batch = torch.from_numpy(np.zeros(num_points)).long()
        y = np.asarray(np.random.randint(0, 2, 5))
        uniq, counts = np.unique(y, return_counts=True)
        self.answer = uniq[np.argmax(counts)]
        y = torch.from_numpy(y)
        self.data = Data(pos=pos, batch=batch, y=y)

    def test_simple(self):
        """
        This test verifies that the class output is correct and corresponds to the maximun vote from sub_part
        """
        out = self.sampler(self.data)
        y = out.y.detach().cpu().numpy()
        npt.assert_array_almost_equal(self.answer, y)

    def loop(self, data, gr, sparse, msg):
        shapes = []
        u = data.clone()
        for i in range(2):
            u = gr(u)
            shapes.append(u.pos.shape[0])

        data = sparse(u)
        num_points = np.unique(data.pos, axis=0).shape[0]

        shapes.append(num_points)
        self.assertEqual(shapes, [shapes[0] for _ in range(len(shapes))])

    def test_double_grid_sampling(self):
        data_random = Data(pos=torch.randn(1000, 3) * 0.1, x=torch.ones((1000, 1)))
        data_fragment = torch.load(os.path.join(DIR_PATH, "test_data/fragment_000003.pt"))

        sparse = cT.GridSampling3D(0.02, quantize_coords=True)
        gr = cT.GridSampling3D(0.02)

        self.loop(data_random, gr, gr, "random")
        self.loop(data_fragment, gr, sparse, "fragment")

    def test_quantize(self):
        data_random = Data(pos=torch.randn(100, 3) * 0.1, x=torch.ones((100, 1)))
        gr = cT.GridSampling3D(0.2, quantize_coords=True)
        quantized = gr(data_random)
        self.assertTrue(hasattr(quantized, "coords"))
        self.assertEqual(quantized.x.shape[0], quantized.coords.shape[0])
        self.assertEqual(quantized.num_nodes, quantized.coords.shape[0])
        self.assertEqual(quantized.coords.dtype, torch.int)


if __name__ == "__main__":
    unittest.main()
