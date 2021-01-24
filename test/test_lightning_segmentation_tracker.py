import unittest
import os
import torch
import sys
from torch_geometric.data import Data

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.metrics.segmentation_tracker import LightningSegmentationTracker

# from torch_points3d.metrics.segmentation_helpers import SegmentationVoter
# from torch_points3d.metrics.classification_tracker import ClassificationTracker
# from torch_points3d.metrics.s3dis_tracker import S3DISTracker


class MockDataset:
    INV_OBJECT_LABEL = {0: "first", 1: "wall", 2: "not", 3: "here", 4: "hoy"}
    pos = torch.tensor([[1, 0, 0], [2, 0, 0], [3, 0, 0], [-1, 0, 0]]).float()
    test_label = torch.tensor([1, 1, 0, 0])

    def __init__(self):
        self.num_classes = 2

    @property
    def test_data(self):
        return Data(pos=self.pos, y=self.test_label)

    def has_labels(self, stage):
        return True


class MockModel:
    def __init__(self):
        self.iter = 0
        self.losses = [
            {"loss_1": 1, "loss_2": 2},
            {"loss_1": 2, "loss_2": 2},
            {"loss_1": 1, "loss_2": 2},
            {"loss_1": 1, "loss_2": 2},
        ]
        self.outputs = [
            torch.tensor([[0, 1], [0, 1]]),
            torch.tensor([[1, 0], [1, 0]]),
            torch.tensor([[1, 0], [1, 0]]),
            torch.tensor([[1, 0], [1, 0], [1, 0]]),
        ]
        self.labels = [torch.tensor([1, 1]), torch.tensor([1, 1]), torch.tensor([1, 1]), torch.tensor([0, 0, -100])]
        self.batch_idx = [torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([0, 0, 1])]

    def get_input(self):
        return Data(pos=MockDataset.pos[:2, :], origin_id=torch.tensor([0, 1]))

    def get_output(self):
        return self.outputs[self.iter].float()

    def get_labels(self):
        return self.labels[self.iter]

    def get_current_losses(self):
        return self.losses[self.iter]

    def get_batch(self):
        return self.batch_idx[self.iter]

    @property
    def device(self):
        return "cpu"


class TestSegmentationTracker(unittest.TestCase):
    def test_forward(self):
        tracker = LightningSegmentationTracker(num_classes=2, stage="train")
        model = MockModel()
        metrics = tracker(model)
        # metrics = tracker.get_metrics()

        for k in ["train_acc", "train_miou", "train_macc"]:
            self.assertAlmostEqual(metrics[k], 100, 5)

        model.iter += 1
        metrics = tracker(model)
        # metrics = tracker.get_metrics()
        metrics = tracker.finalise()
        for k in ["train_acc", "train_macc"]:
            self.assertEqual(metrics[k], 50)
        self.assertAlmostEqual(metrics["train_miou"], 25, 5)
        self.assertEqual(metrics["train_loss_1"], 1.5)

        tracker.reset("test")
        model.iter += 1
        metrics = tracker(model)
        # metrics = tracker.get_metrics()
        for k in ["test_acc", "test_miou", "test_macc"]:
            self.assertAlmostEqual(metrics[k].item(), 0, 5)

    def test_ignore_label(self):
        tracker = LightningSegmentationTracker(num_classes=2, ignore_label=-100)
        tracker.reset("test")
        model = MockModel()
        model.iter = 3
        metrics = tracker(model)
        # metrics = tracker.get_metrics()
        for k in ["test_acc", "test_miou", "test_macc"]:
            self.assertAlmostEqual(metrics[k], 100, 5)

    def test_finalise(self):
        tracker = LightningSegmentationTracker(num_classes=2, ignore_label=-100)
        tracker.reset("test")
        model = MockModel()
        model.iter = 3
        tracker(model)
        tracker.finalise()
        with self.assertRaises(RuntimeError):
            tracker(model)


if __name__ == "__main__":
    unittest.main()
