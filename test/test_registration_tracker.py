import unittest
import numpy as np
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)

from src.metrics.registration_tracker import FragmentRegistrationTracker


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
        self.xyz = 0

        self.batch_idx = [np.asarray([0, 1]), np.asarray([0, 1]), np.asarray([0, 1]), np.asarray([0, 0, 1])]

    def get_outputs(self):
        return self.outputs[self.iter]

    def get_xyz(self):
        pass

    def get_ind(self):
        pass

    def get_current_losses(self):
        return self.losses[self.iter]

    def get_batch_idx(self):
        return self.batch_idx[self.iter]


class TestSegmentationTracker(unittest.TestCase):
    def test_track(self):
        tracker = FragmentRegistationTracker(MockDataset())
        model = MockModel()
        tracker.track(model)
        metrics = tracker.get_metrics()
        list_variables = ["hit_ratio", "feat_match_ratio", "trans_error", "rot_error"]
        list_test_variables = ["test_" + f for f in list_variables]
        for k in list_test_variables:
            self.assertAlmostEqual(metrics[k], 100, 5)

        model.iter += 1
        tracker.track(model)
        metrics = tracker.get_metrics()
        for k in ["train_acc", "train_macc"]:
            self.assertEqual(metrics[k], 50)
        self.assertAlmostEqual(metrics["train_miou"], 25, 5)
        self.assertEqual(metrics["train_loss_1"], 1.5)

        tracker.reset("test")
        model.iter += 1
        tracker.track(model)
        metrics = tracker.get_metrics()
        for k in ["test_acc", "test_miou", "test_macc"]:
            self.assertAlmostEqual(metrics[k], 0, 5)


if __name__ == "__main__":
    unittest.main()
