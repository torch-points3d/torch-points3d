import unittest
import os
import torch
import sys
import numpy as np
import numpy.testing as npt
from torch_geometric.data import Data

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.metrics.segmentation_helpers import SegmentationVoter
from torch_points3d.metrics.classification_tracker import ClassificationTracker
from torch_points3d.metrics.s3dis_tracker import S3DISTracker


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
    def test_track(self):
        tracker = SegmentationTracker(MockDataset())
        model = MockModel()
        tracker.track(model)
        metrics = tracker.get_metrics()
        for k in ["train_acc", "train_miou", "train_macc"]:
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

    def test_ignore_label(self):
        tracker = SegmentationTracker(MockDataset(), ignore_label=-100)
        tracker.reset("test")
        model = MockModel()
        model.iter = 3
        tracker.track(model)
        metrics = tracker.get_metrics()
        for k in ["test_acc", "test_miou", "test_macc"]:
            self.assertAlmostEqual(metrics[k], 100, 5)

    def test_finalise(self):
        tracker = SegmentationTracker(MockDataset(), ignore_label=-100)
        tracker.reset("test")
        model = MockModel()
        model.iter = 3
        tracker.track(model)
        tracker.finalise()
        with self.assertRaises(RuntimeError):
            tracker.track(model)

    def test_seg_full_res_helpers(self):

        raw_pos = torch.from_numpy(np.asarray([[0, 0, 0], [0, 0.5, 0], [0.5, 1, 0], [1, 1, 0]]))
        raw_y = torch.from_numpy(np.asarray([0, 0, 1, 1]))
        preds = torch.from_numpy(np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]]))
        idx = torch.arange(0, 4)
        raw_data = Data(pos=raw_pos, y=raw_y)

        np.asarray([1, 1, 0, 0])
        left_pred = np.asarray([0, 0, 0, 0])
        np.asarray([0, 0, 1, 1])
        right_pred = np.asarray([1, 1, 1, 1])

        for _ in range(25):
            segmentation_resolver = SegmentationVoter(raw_data, 2, "dense")

            for _ in range(np.random.randint(1, 10)):
                slice_ = np.random.choice(range(4), 2)
                data = Data(pos=raw_pos[slice_], y=raw_y[slice_])
                setattr(data, SaveOriginalPosId.KEY, [idx[slice_]])
                output = preds[slice_]
                segmentation_resolver.add_vote(data, output, 0)

            mask = segmentation_resolver._vote_counts.numpy() > 0
            if np.sum(mask > 0) > 2:
                npt.assert_array_almost_equal(segmentation_resolver.full_res_preds.numpy(), raw_y)
            else:
                has_left = np.sum(mask[:2]) > 0
                has_right = np.sum(mask[2:]) > 0

                if has_left and has_right:
                    npt.assert_array_almost_equal(segmentation_resolver.full_res_preds.numpy(), raw_y)

                elif has_left and not has_right:
                    npt.assert_array_almost_equal(segmentation_resolver.full_res_preds.numpy(), left_pred)
                else:
                    npt.assert_array_almost_equal(segmentation_resolver.full_res_preds.numpy(), right_pred)

        segmentation_resolver.k = 5
        self.assertEqual(segmentation_resolver.k, 5)


class TestS3DISTarcker(unittest.TestCase):
    def test_fullres(self):
        tracker = S3DISTracker(MockDataset())
        tracker.reset("test")
        model = MockModel()
        tracker.track(model, full_res=True)
        tracker.track(model, full_res=True)
        tracker.finalise(full_res=True)
        metrics = tracker.get_metrics(verbose=True)
        self.assertAlmostEqual(metrics["test_full_vote_miou"], 25, 5)
        self.assertAlmostEqual(metrics["test_vote_miou"], 100, 5)


class TestClassificationTracker(unittest.TestCase):
    from torch_points3d.metrics.classification_tracker import ClassificationTracker

    def test_classification(self):
        tracker = ClassificationTracker(MockDataset())
        tracker.reset("test")
        model = MockModel()
        tracker.track(model)
        metrics = tracker.get_metrics()
        self.assertAlmostEqual(metrics["test_acc"], 100, 5)

        model.iter += 1
        tracker.track(model)
        metrics = tracker.get_metrics()
        self.assertAlmostEqual(metrics["test_acc"], 50, 5)


if __name__ == "__main__":
    unittest.main()
