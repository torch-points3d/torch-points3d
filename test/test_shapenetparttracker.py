import unittest
import numpy as np
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)

from torch_points3d.metrics.shapenet_part_tracker import ShapenetPartTracker


class MockDataset:
    def __init__(self):
        self.num_classes = 5
        self.class_to_segments = {"class1": [0, 1], "class2": [2, 3]}
        self.is_hierarchical = True


class MockModel:
    def __init__(self):
        self.iter = 0
        self.losses = [{"loss_1": 1, "loss_2": 2}, {"loss_1": 1, "loss_2": 2}, {"loss_1": 1, "loss_2": 2}]
        self.outputs = [
            np.asarray([[0, 1, 0, 0], [0, 1, 0, 0]]),
            np.asarray([[0, 0, 1, 0], [0, 0, 1, 0]]),
            np.asarray([[0, 0, 1, 0]]),
        ]
        self.labels = [np.asarray([1, 1]), np.asarray([2, 2]), np.asarray([3])]
        self.batch_idx = [np.asarray([0, 1]), np.asarray([0, 1]), np.asarray([0])]
        self.conv_type = "DENSE"

    def get_output(self):
        return self.outputs[self.iter]

    def get_labels(self):
        return self.labels[self.iter]

    def get_current_losses(self):
        return self.losses[self.iter]

    def get_batch(self):
        return self.batch_idx[self.iter]


class TestSegmentationTracker(unittest.TestCase):
    def test_track(self):
        tracker = ShapenetPartTracker(MockDataset())
        model = MockModel()
        tracker.track(model)
        metrics = tracker.get_metrics(verbose=True)
        for k in ["train_Cmiou", "train_Imiou"]:
            self.assertAlmostEqual(metrics[k], 100, 5)

        model.iter += 1
        tracker.track(model)
        metrics = tracker.get_metrics(verbose=True)
        for k in ["train_Cmiou", "train_Imiou"]:
            self.assertAlmostEqual(metrics[k], 100, 5)

        model.iter += 1
        tracker.track(model)
        metrics = tracker.get_metrics(verbose=True)
        self.assertAlmostEqual(metrics["train_Imiou"], 4 * 100 / 5)
        self.assertAlmostEqual(metrics["train_Cmiou"], (100 + 200 / 3.0) / 2.0)
        # for k in ["train_Cmiou", "train_Imiou"]:
        #     self.assertAlmostEqual(metrics[k], 100, 5)


if __name__ == "__main__":
    unittest.main()
