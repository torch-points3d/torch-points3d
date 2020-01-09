import unittest
import torch
import numpy as np
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)

from metrics.metrics_tracker import SegmentationTracker


class MockDataset:
    def __init__(self):
        self.num_classes = 2


class TestSegmentationTracker(unittest.TestCase):
    def test_track(self):
        tracker = SegmentationTracker(MockDataset())
        tracker.track({"loss_1": 1, "loss_2": 2}, np.asarray([[0, 1], [0, 1]]), np.asarray([1, 1]))
        metrics = tracker.get_metrics()
        for k in ["train_acc", "train_miou", "train_macc", "train_acc"]:
            self.assertEqual(metrics[k], 100)

        tracker.track({"loss_1": 2, "loss_2": 2}, np.asarray([[1, 0], [1, 0]]), np.asarray([1, 1]))
        metrics = tracker.get_metrics()
        for k in ["train_acc", "train_miou", "train_macc", "train_acc"]:
            self.assertEqual(metrics[k], 50)
        self.assertEqual(metrics["train_loss_1"], 1.5)

        tracker.reset("test")
        tracker.track({"loss_1": 2, "loss_2": 2}, np.asarray([[1, 0], [1, 0]]), np.asarray([1, 1]))
        metrics = tracker.get_metrics()
        for k in ["test_acc", "test_miou", "test_macc", "test_acc"]:
            self.assertEqual(metrics[k], 0)


if __name__ == "__main__":
    unittest.main()
