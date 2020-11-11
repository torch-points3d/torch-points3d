import numpy as np
import torch
import unittest
import numpy.testing as npt
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points3d.utils.registration import estimate_transfo
from torch_points3d.utils.registration import fast_global_registration
from torch_points3d.utils.registration import ransac_registration
from torch_points3d.metrics.registration_metrics import compute_hit_ratio
from torch_points3d.metrics.registration_metrics import compute_transfo_error
from torch_points3d.metrics.registration_metrics import compute_scaled_registration_error
from torch_points3d.utils.geometry import rodrigues
from torch_points3d.utils.geometry import euler_angles_to_rotation_matrix


class TestRegistrationMetrics(unittest.TestCase):
    def test_estimate_transfo(self):

        a = torch.randn(100, 3)

        R_gt = euler_angles_to_rotation_matrix(torch.rand(3) * np.pi)
        t_gt = torch.rand(3)
        T_gt = torch.eye(4)
        T_gt[:3, :3] = R_gt
        T_gt[:3, 3] = t_gt
        b = a.mm(R_gt.T) + t_gt
        T_pred = estimate_transfo(a, b)

        npt.assert_allclose(T_pred.numpy(), T_gt.numpy(), rtol=1e-3)

    def test_fast_global_registration(self):
        a = torch.randn(100, 3)

        R_gt = euler_angles_to_rotation_matrix(torch.rand(3) * np.pi)
        t_gt = torch.rand(3)
        T_gt = torch.eye(4)
        T_gt[:3, :3] = R_gt
        T_gt[:3, 3] = t_gt
        b = a.mm(R_gt.T) + t_gt
        T_pred = fast_global_registration(a, b, mu_init=1, num_iter=20)
        npt.assert_allclose(T_pred.numpy(), T_gt.numpy(), rtol=1e-3)

    def test_fast_global_registration_with_outliers(self):
        a = torch.randn(100, 3)
        R_gt = euler_angles_to_rotation_matrix(torch.rand(3) * np.pi)
        t_gt = torch.rand(3)
        T_gt = torch.eye(4)
        T_gt[:3, :3] = R_gt
        T_gt[:3, 3] = t_gt
        b = a.mm(R_gt.T) + t_gt
        b[[1, 5, 20, 32, 74, 17, 27, 77, 88, 89]] *= 42
        T_pred = fast_global_registration(a, b, mu_init=1, num_iter=20)
        # T_pred = estimate_transfo(a, b)
        npt.assert_allclose(T_pred.numpy(), T_gt.numpy(), rtol=1e-3)

    def test_ransac(self):
        a = torch.randn(100, 3)
        R_gt = euler_angles_to_rotation_matrix(torch.rand(3) * np.pi)
        t_gt = torch.rand(3)
        T_gt = torch.eye(4)
        T_gt[:3, :3] = R_gt
        T_gt[:3, 3] = t_gt
        b = a.mm(R_gt.T) + t_gt
        b[[1, 5, 20, 32, 74, 17, 27, 77, 88, 89]] *= 42
        T_pred = ransac_registration(a, b, distance_threshold=0.01)
        # T_pred = estimate_transfo(a, b)
        npt.assert_allclose(T_pred.numpy(), T_gt.numpy(), rtol=1e-3)

    def test_compute_hit_ratio(self):
        xyz = torch.randn(100, 3)
        xyz_target = xyz.clone()
        xyz[[1, 5, 20, 32, 74, 17, 27, 77, 88, 89]] += 42

        hit = compute_hit_ratio(xyz, xyz_target, torch.eye(4), 0.1)

        self.assertAlmostEqual(hit.item(), 0.9)

    def test_compute_transfo_error(self):

        axis = torch.randn(3)
        axis = axis / torch.norm(axis)
        theta = 30 * np.pi / 180

        R = rodrigues(axis, theta)

        T = torch.eye(4)
        T[:3, :3] = R
        T[0, 3] = 1

        rte, rre = compute_transfo_error(torch.eye(4), T)
        npt.assert_allclose(rte.item(), 1)
        npt.assert_allclose(rre.item(), 30, rtol=1e-3)

    def test_compute_scaled_registration_error(self):

        xyz = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 0.0, 0.0],])
        R_est = euler_angles_to_rotation_matrix(torch.tensor([0, 0, np.pi / 6]))
        T_gt = torch.eye(4)
        T_est = torch.eye(4)
        T_est[:3, :3] = R_est
        err = compute_scaled_registration_error(xyz, T_gt, T_est)
        val = 0.55901
        self.assertAlmostEqual(err.item(), val, places=5)


if __name__ == "__main__":
    unittest.main()
