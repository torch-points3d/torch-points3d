import unittest
import torch
import numpy as np
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)

from metrics.hierarchical_segmentation_tracker import HierarchicalSegmentationTracker


class MockDataset:
    def __init__(self):
        self.num_classes = 5
        self.class_to_segments = {"class1": [0, 1], "class2": [2, 3, 4]}
        self.is_hierarchical = True


class MockModel:
    def __init__(self):
        self.iter = 0
        self.losses = [{"loss_1": 1, "loss_2": 2}, {"loss_1": 1, "loss_2": 2}, {"loss_1": 1, "loss_2": 2}]
        self.outputs = [
            np.asarray([[0, 1, 0], [0, 1, 0]]),
            np.asarray([[0, 0, 1], [0, 0, 1]]),
            np.asarray([[0, 0, 1, 0]]),
        ]
        self.labels = [np.asarray([1, 1]), np.asarray([2, 2]), np.asarray([3])]
        self.batch_idx = [np.asarray([0, 1]), np.asarray([0, 1]), np.asarray([0])]

    def get_output(self):
        return self.outputs[self.iter]

    def get_labels(self):
        return self.labels[self.iter]

    def get_current_losses(self):
        return self.losses[self.iter]

    def get_batch_idx(self):
        return self.batch_idx[self.iter]


class TestSegmentationTracker(unittest.TestCase):
    def test_track(self):
        tracker = HierarchicalSegmentationTracker(MockDataset())
        model = MockModel()
        tracker.track(model)
        metrics = tracker.get_metrics(verbose=True)
        for k in ["train_acc", "train_miou", "train_macc", "train_acc"]:
            self.assertEqual(metrics[k], 100)

        model.iter += 1
        tracker.track(model)
        metrics = tracker.get_metrics(verbose=True)
        for k in ["train_acc", "train_miou", "train_macc", "train_acc"]:
            self.assertEqual(metrics[k], 100)

        model.iter += 1
        tracker.track(model)
        metrics = tracker.get_metrics(verbose=True)
        self.assertEqual(metrics["train_macc_per_class"]["class2"], 50)
        self.assertEqual(int(metrics["train_acc_per_class"]["class2"]), 66)
        self.assertEqual(int(metrics["train_miou_per_class"]["class2"]), 33)
        self.assertEqual(int(metrics["train_Cmiou"]), int((100 + 33) / 2))


if __name__ == "__main__":
    unittest.main()
