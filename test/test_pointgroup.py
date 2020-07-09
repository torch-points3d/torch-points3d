import os
import sys
import unittest
import torch

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points3d.core.losses import offset_loss, instance_iou_loss


class TestPointGroupLosses(unittest.TestCase):
    def test_offset_loss(self):
        pred_offset = torch.tensor([[2, 0, 0], [0, 1, 0]]).float()
        gt_offsets = torch.tensor([[2, 0, 0], [0, 1, 0]]).float()

        losses = offset_loss(pred_offset, gt_offsets, 2)
        self.assertEqual(losses["offset_norm_loss"].item(), 0)
        self.assertAlmostEqual(losses["offset_dir_loss"].item(), -1, places=5)

        gt_offsets = torch.tensor([[2, 0, 0], [0, -1, 0]]).float()

        losses = offset_loss(pred_offset, gt_offsets, 2)
        self.assertAlmostEqual(losses["offset_norm_loss"].item(), (0 + 2) / 2.0, places=5)
        self.assertAlmostEqual(losses["offset_dir_loss"].item(), (-1 + 1) / 2.0, places=5)

    def test_scoreloss(self):
        clusters = [torch.tensor([0, 1, 2]), torch.tensor([3, 4])]
        scores = torch.tensor([1, 0]).float()
        gt_instances = torch.tensor([1, 1, 1, 0, 0])
        batch = torch.tensor([0, 0, 0, 0, 0])
        loss = instance_iou_loss(clusters, scores, gt_instances, batch)

        self.assertEqual(loss.item(), 0)

        gt_instances = torch.tensor([1, 1, 1, 2, 2])
        loss = instance_iou_loss(clusters, scores, gt_instances, batch)

        self.assertAlmostEqual(loss.item(), 50)


if __name__ == "__main__":
    unittest.main()
