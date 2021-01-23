"""
Test tracker
"""

import unittest
import torch

from torch_points3d.metrics.classification_tracker import LightningClassificationTracker


class MockModel:
    def __init__(self):
        self.ind = 0

    def get_output(self):

        if self.ind == 0:
            return torch.tensor([1.0, 0.0])
        elif self.ind == 1:
            return torch.tensor([1.0, 0.0])
        elif self.ind == 2:
            return torch.tensor([0.0, 1.0])
        elif self.ind == 3:
            return torch.tensor([1.0, 0.0])

    def get_labels(self):
        if self.ind == 0:
            return torch.tensor([0])
        elif self.ind == 1:
            return torch.tensor([0])
        elif self.ind == 2:
            return torch.tensor([0])
        elif self.ind == 3:

            return torch.tensor([0])


class TestClassificationTracker(unittest.TestCase):
    def test_forward(self):
        model = MockModel()
        tracker = LightningClassificationTracker()
        metric = tracker(model)
        self.assertAlmostEqual(metric["train_acc"].item(), 1.0, 5)

    def test_finalise(self):
        model = MockModel()
        tracker = LightningClassificationTracker()

        for i in range(4):
            model.ind = i
            tracker(model)
        final_metric = tracker.finalize()
        self.assertAlmostEqual(final_metric["train_acc"].item(), 0.75, 5)


if __name__ == "__main__":
    unittest.main()
