import unittest
import torch
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)
from torch_points3d.core.losses.dirichlet_loss import (
    _variance_estimator_dense,
    dirichlet_loss,
    _variance_estimator_sparse,
)


class TestDirichletLoss(unittest.TestCase):
    def test_loss_dense(self):
        pos = torch.tensor([[[0, 0, 0], [1, 0, 0], [1.1, 0, 0]]], dtype=torch.float)
        f = torch.tensor([[1, 1, 3]], dtype=torch.float)

        var = _variance_estimator_dense(1.01, pos, f)
        torch.testing.assert_allclose(var, [[0, 4, 4]])

        loss = dirichlet_loss(1.01, pos, f)
        self.assertAlmostEqual(loss.item(), 4 / 3.0)

    def test_loss_sparse(self):
        pos = torch.tensor([[0, 0, 0], [1, 0, 0], [1.1, 0, 0], [0, 0, 0], [1, 0, 0], [1.1, 0, 0]], dtype=torch.float)
        f = torch.tensor([1, 1, 3, 0, 1, 0], dtype=torch.float)
        batch_idx = torch.tensor([0, 0, 0, 1, 1, 1])

        var = _variance_estimator_sparse(1.01, pos, f, batch_idx)
        torch.testing.assert_allclose(var, [0, 4, 4, 1, 2, 1])

        loss = dirichlet_loss(1.01, pos, f, batch_idx)
        self.assertAlmostEqual(loss.item(), sum([0, 4, 4, 1, 2, 1]) / (2 * 6))


if __name__ == "__main__":
    unittest.main()
