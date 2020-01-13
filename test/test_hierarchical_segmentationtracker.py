import unittest
import torch
import numpy as np
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)

from metrics.hierachical_segmentation_tracker import HierachicalSegmentationTracker


class MockDataset:
    def __init__(self):
        self.num_classes = 5
        self.class_to_segments = {"class1": [0, 1], "class2": [2, 3, 4]}
        self.is_hierarchical = True


class TestSegmentationTracker(unittest.TestCase):
    def test_track(self):
        tracker = HierachicalSegmentationTracker(MockDataset())
        tracker.track({"loss_1": 1, "loss_2": 2}, np.asarray([[0, 1, 0], [0, 1, 0]]), np.asarray([1, 1]))
        metrics = tracker.get_metrics(verbose=True)
        for k in ["train_acc", "train_miou", "train_macc", "train_acc"]:
            self.assertEqual(metrics[k], 100)

        tracker.track({"loss_1": 1, "loss_2": 2}, np.asarray([[0, 0, 1], [0, 0, 1]]), np.asarray([2, 2]))
        metrics = tracker.get_metrics(verbose=True)
        for k in ["train_acc", "train_miou", "train_macc", "train_acc"]:
            self.assertEqual(metrics[k], 100)

        tracker.track({"loss_1": 1, "loss_2": 2}, np.asarray([[0, 0, 1, 0]]), np.asarray([3]))
        metrics = tracker.get_metrics(verbose=True)
        self.assertEqual(metrics["train_macc_per_class"]["class2"], 50)
        self.assertEqual(int(metrics["train_acc_per_class"]["class2"]), 66)
        self.assertEqual(int(metrics["train_miou_per_class"]["class2"]), 33)
        self.assertEqual(int(metrics["train_miou"]), int((100 + 33) / 2))


if __name__ == "__main__":
    unittest.main()
