import os
import sys
import unittest
import numpy as np
import numpy.testing as npt
import torch

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points3d.modules.KPConv.losses import repulsion_loss, fitting_loss, permissive_loss


class TestKPConvLosses(unittest.TestCase):
    def test_permissive_loss(self):
        pos_n = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]).astype(np.float)
        pos_t = torch.from_numpy(pos_n)
        loss = permissive_loss(pos_t, 1).item()
        assert loss == np.sqrt(2)

    def test_fitting_loss(self):
        pos_n = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]).astype(np.float)
        target = np.asarray([[0.5, 0.5, 0]])
        K_points = torch.from_numpy(pos_n)
        neighbors = torch.from_numpy(target)
        neighbors = neighbors
        neighbors = neighbors.repeat([4, 1])
        differences = neighbors - K_points
        sq_distances = torch.sum(differences ** 2, dim=-1).unsqueeze(0)
        loss = fitting_loss(sq_distances, 1).item()
        assert loss == 0.5

    def test_repulsion_loss(self):
        pos_n = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]).astype(np.float64)
        K_points = torch.from_numpy(pos_n)
        loss = repulsion_loss(K_points.unsqueeze(0), 1).item()
        arr_ = np.asarray([0.25, 0.25, 0.0074]).astype(np.float64)
        # Pytorch losses precision from decimal 4
        npt.assert_almost_equal(loss, 4 * np.sum(arr_), decimal=3)


if __name__ == "__main__":
    unittest.main()
