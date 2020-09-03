import unittest
import sys
import os

import torch
from torch_geometric.data import Batch

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, ".."))
torch.manual_seed(0)

from torch_points3d.core.geometry.se3 import SE3Transform
from torch_points3d.utils.geometry import euler_angles_to_rotation_matrix


class TestHelpers(unittest.TestCase):
    def test_batch_transform(self):
        num_pt = 10
        pos = torch.arange(num_pt * 3).reshape(num_pt, 3).expand(3, num_pt, 3).float()
        data = Batch(pos=pos)
        trans = torch.eye(4).expand(3, 4, 4)
        trans[1, :3, 3] = torch.tensor([10.0, 12.0, -5.0])
        trans[2, :3, :3] = torch.tensor(
            [[0.8138, -0.4410, 0.3785], [0.4698, 0.8826, 0.0180], [-0.3420, 0.1632, 0.9254]]
        )
        transform = SE3Transform(conv_type="DENSE")
        new_data = transform(trans, data.clone())
        for i in range(3):
            torch.testing.assert_allclose(new_data.pos[i], data.pos[i] @ trans[i][:3, :3].T + trans[i][:3, 3])

    def test_multi_batch_transform(self):

        # create 3 x 6 transformations
        num_pt = 100
        trans = torch.eye(4).expand(3, 4, 4).expand(6, 3, 4, 4)
        theta = torch.tensor(
            [
                [
                    30 * (i // 3 + 1) * (i % 3 == 0),
                    30 * (i // 3 + 1) * ((i - 1) % 3 == 0),
                    30 * (i // 3 + 1) * ((i - 2) % 3 == 0),
                ]
                for i in range(18)
            ]
        ).float()

        for b in range(3):
            for j in range(6):
                trans[j, b][:3, :3] = euler_angles_to_rotation_matrix(theta[j + b * 6])
        pos = torch.arange(num_pt * 3).reshape(num_pt, 3).expand(3, num_pt, 3).float()
        # [B, N, 3]
        data = Batch(pos=pos)
        transform = SE3Transform(conv_type="DENSE")
        new_data = transform(trans, data.clone())  # [6, B, N, 3]
        for b in range(3):
            for j in range(6):
                gt_pos = data.pos[b] @ trans[j, b][:3, :3].T + trans[j, b][:3, 3]
                torch.testing.assert_allclose(new_data.pos[j + b * 6], gt_pos)

    def test_partial_transform(self):
        num_pt = 100
        pos = torch.arange(num_pt * 3).reshape(num_pt, 3).float()
        batch = torch.randint(3, (num_pt,)).sort()[0].long()
        data = Batch(pos=pos, batch=batch)
        trans = torch.eye(4).expand(3, 4, 4)
        trans[1, :3, 3] = torch.tensor([10.0, 12.0, -5.0])
        trans[2, :3, :3] = torch.tensor(
            [[0.8138, -0.4410, 0.3785], [0.4698, 0.8826, 0.0180], [-0.3420, 0.1632, 0.9254]]
        )
        transform = SE3Transform(conv_type="PARTIAL")
        new_data = transform(trans, data.clone())
        for i in range(3):
            torch.testing.assert_allclose(
                new_data.pos[batch == i], data.pos[batch == i] @ trans[i][:3, :3].T + trans[i][:3, 3]
            )

    def test_multi_partial_transform(self):
        num_pt = 100
        trans = torch.eye(4).expand(3, 4, 4).expand(6, 3, 4, 4)
        theta = torch.tensor(
            [
                [
                    30 * (i // 3 + 1) * (i % 3 == 0),
                    30 * (i // 3 + 1) * ((i - 1) % 3 == 0),
                    30 * (i // 3 + 1) * ((i - 2) % 3 == 0),
                ]
                for i in range(18)
            ]
        ).float()

        for b in range(3):
            for j in range(6):
                trans[j, b][:3, :3] = euler_angles_to_rotation_matrix(theta[j + b * 6])
        pos = torch.arange(num_pt * 3).reshape(num_pt, 3).float()
        batch = torch.randint(3, (num_pt,)).sort()[0].long()
        data = Batch(pos=pos, batch=batch)
        transform = SE3Transform(conv_type="PARTIAL")
        new_data = transform(trans, data.clone())
        for j in range(6):
            for b in range(3):
                gt_pos = data.pos[batch == b] @ trans[j, b][:3, :3].T + trans[j, b][:3, 3]
                torch.testing.assert_allclose(new_data.pos[new_data.batch == (b + j * 3)], gt_pos)

    def test_multiscale_partial_transform(self):
        # data where multiscale have been computed
        pass

    def test_sparse_transform(self):
        # data are sparse tensors
        pass


if __name__ == "__main__":
    unittest.main()
