import unittest
import torch
import numpy as np
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)

from metrics.segmentation_tracker import SegmentationTracker


class MockDataset:
    def __init__(self):
        self.num_classes = 2


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
            np.asarray([[0, 1], [0, 1]]),
            np.asarray([[1, 0], [1, 0]]),
            np.asarray([[1, 0], [1, 0]]),
            np.asarray([[1, 0], [1, 0], [1, 0]]),
        ]
        self.labels = [np.asarray([1, 1]), np.asarray([1, 1]), np.asarray([1, 1]), np.asarray([1, 1, 0])]
        self.batch_idx = [np.asarray([0, 1]), np.asarray([0, 1]), np.asarray([0, 1]), np.asarray([0, 0, 1])]

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
        tracker = SegmentationTracker(MockDataset())
        model = MockModel()
        tracker.track(model)
        metrics = tracker.get_metrics()
        for k in ["train_acc", "train_miou", "train_macc", "train_acc"]:
            self.assertEqual(metrics[k], 100)

        model.iter += 1
        tracker.track(model)
        metrics = tracker.get_metrics()
        for k in ["train_acc", "train_macc", "train_acc"]:
            self.assertEqual(metrics[k], 50)
        self.assertEqual(metrics["train_miou"], 25)
        self.assertEqual(metrics["train_loss_1"], 1.5)

        tracker.reset("test")
        model.iter += 1
        tracker.track(model)
        metrics = tracker.get_metrics()
        for k in ["test_acc", "test_miou", "test_macc", "test_acc"]:
            self.assertEqual(metrics[k], 0)

    def test_instance(self):
        tracker = SegmentationTracker(MockDataset())
        model = MockModel()
        model.iter = 3
        tracker.track(model)
        metrics = tracker.get_metrics(verbose=True)
        for k in ["train_Iacc", "train_Imiou", "train_Imacc"]:
            self.assertEqual(metrics[k], 50)


if __name__ == "__main__":
    unittest.main()
