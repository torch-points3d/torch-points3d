import numpy as np
import torch
import unittest
import numpy.testing as npt

from src.metrics.registration_metrics import estimate_transfo
from src.metrics.registration_metrics import get_matches
from src.metrics.registration_metrics import fast_global_registration
from src.metrics.registration_metrics import compute_hit_ratio
from src.metrics.registration_metrics import compute_transfo_error
from src.core.data_transform import euler_angles_to_rotation_matrix


class TestRegistrationMetrics(unittest.TestCase):
    def test_estimate_transfo(self):

        a = torch.randn(100, 3)

        R_gt = euler_angles_to_rotation_matrix(torch.rand(3) * np.pi)
        t_gt = torch.rand(3)
        T_gt = torch.eye(4)
        T_gt[:3, :3] = R_gt
        T_gt[:3, 3] = t_gt
        b = a.mm(R_gt.T) + t_gt
        match = torch.stack([torch.arange(0, len(a)), torch.arange(0, len(a))]).T

        T_pred = estimate_transfo(a, b, match)

        npt.assert_allclose(T_pred.numpy(), T_gt.numpy(), rtol=1e-3)

    def test_get_matches(self):
        feat = torch.tensor([[1, 0, 0], [0, 0, 1], [0.436, 0.9, 0]])
        feat_target = torch.tensor([[-0.01, 0.01, 0.999], [0.42, 0.89, -0.12], [0.98, -0.17, 0.1]])
        match_gt = torch.tensor([[0, 2], [1, 0], [2, 1]])
        match_pred, _ = get_matches(feat, feat_target)
        npt.assert_array_almost_equal(match_pred.numpy(), match_gt.numpy())

    def test_fast_global_registration(self):
        a = torch.randn(100, 3)

        R_gt = euler_angles_to_rotation_matrix(torch.rand(3) * np.pi)
        t_gt = torch.rand(3)
        T_gt = torch.eye(4)
        T_gt[:3, :3] = R_gt
        T_gt[:3, 3] = t_gt
        b = a.mm(R_gt.T) + t_gt
        match = torch.stack([torch.arange(0, len(a)), torch.arange(0, len(a))]).T
        T_pred = fast_global_registration(a, b, match)
        npt.assert_allclose(T_pred.numpy(), T_gt.numpy(), rtol=1e-3)

    def test_compute_hit_ratio(self):
        xyz = torch.randn(100, 3)
        xyz_target = xyz.clone()
        xyz[[1, 5, 20, 32, 74, 17, 27, 77, 88, 89]] += 42

        match = torch.stack([torch.arange(0, len(a)), torch.arange(0, len(a))]).T

        hit = compute_hit_ratio(xyz, xyz_target, match, torch.eye(4), 0.1)

        self.assertAlmostEqual(hit, 0.9)
