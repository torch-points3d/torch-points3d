import os
import sys
import unittest
import torch
import numpy as np
from torch_geometric.data import Data

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points3d.models.panoptic.structures import PanopticResults, PanopticLabels
from torch_points3d.metrics.panoptic_tracker import _Instance, InstanceAPMeter, PanopticTracker


class TestInstance(unittest.TestCase):
    def test_iou(self):
        i1 = _Instance(classname=1, indices=np.array([1, 2, 3]), score=1, scan_id=0)
        i2 = _Instance(classname=1, indices=np.array([1, 2, 3]), score=1, scan_id=0)
        self.assertEqual(i1.iou(i2), 1)

        i2 = _Instance(classname=1, indices=np.array([1, 4]), score=1, scan_id=0)
        self.assertEqual(i1.iou(i2), 0.25)


class TestInstanceAPMeter(unittest.TestCase):
    def test_add(self):
        gts = [
            _Instance(classname=1, indices=np.array([1, 2, 3]), score=1, scan_id=0),
            _Instance(classname=1, indices=np.array([1, 2, 3]), score=1, scan_id=0),
            _Instance(classname=1, indices=np.array([1, 2, 3]), score=1, scan_id=1),
        ]

        preds = [
            _Instance(classname=1, indices=np.array([1, 2, 3]), score=1, scan_id=0),
            _Instance(classname=1, indices=np.array([1, 2, 3]), score=0.1, scan_id=10),
            _Instance(classname=10, indices=np.array([1, 2, 3]), score=0.1, scan_id=10),
        ]

        meter = InstanceAPMeter()
        meter.add(preds, gts)

        self.assertEqual(len(meter._pred_clusters[1]), 2)
        self.assertEqual(meter._pred_clusters[1][0].score, 1)

        self.assertEqual(list(meter._gt_clusters[1].keys()), [0, 1])
        self.assertEqual(len(meter._gt_clusters[1][0]), 2)

    def test_eval_single_class(self):
        gts = [
            _Instance(classname=1, indices=np.array([1, 2, 3]), score=1, scan_id=0),
            _Instance(classname=1, indices=np.array([4, 5]), score=1, scan_id=0),
            _Instance(classname=1, indices=np.array([6, 7, 8]), score=1, scan_id=0),
        ]

        preds = [
            _Instance(classname=1, indices=np.array([1, 2, 3]), score=1, scan_id=0),
            _Instance(classname=1, indices=np.array([6]), score=0, scan_id=0),
        ]
        meter = InstanceAPMeter()
        meter.add(preds, gts)

        rec, prec, ap = meter.eval(0.5)
        np.testing.assert_allclose(rec[1], np.asarray([1.0 / 3.0, 1.0 / 3.0]))
        np.testing.assert_allclose(prec[1], np.asarray([1.0, 1.0 / 2.0]))
        self.assertAlmostEqual(ap[1], 1.0 / 3.0)

    def test_eval_overlap(self):
        gts = [
            _Instance(classname=1, indices=np.array([1, 2, 3]), score=1, scan_id=0),
        ]

        preds = [
            _Instance(classname=1, indices=np.array([1, 2]), score=0, scan_id=0),
            _Instance(classname=1, indices=np.array([2, 3]), score=1, scan_id=0),
        ]
        meter = InstanceAPMeter()
        meter.add(preds, gts)

        rec, prec, _ = meter.eval(0.5)
        np.testing.assert_allclose(rec[1], np.asarray([1.0, 1.0]))
        np.testing.assert_allclose(prec[1], np.asarray([1.0, 0.5]))

    def test_eval_two_classes(self):
        gts = [
            _Instance(classname=1, indices=np.array([1, 2, 3]), score=1, scan_id=0),
            _Instance(classname=1, indices=np.array([4, 5]), score=1, scan_id=0),
            _Instance(classname=2, indices=np.array([6, 7]), score=1, scan_id=0),
        ]

        preds = [
            _Instance(classname=1, indices=np.array([1, 2, 3]), score=1, scan_id=0),
            _Instance(classname=2, indices=np.array([6]), score=0, scan_id=0),
        ]
        meter = InstanceAPMeter()
        meter.add(preds, gts)

        rec, prec, _ = meter.eval(0.25)
        np.testing.assert_allclose(rec[1], np.asarray([1.0 / 2.0]))
        np.testing.assert_allclose(prec[1], np.asarray([1.0]))

        np.testing.assert_allclose(rec[2], np.asarray([1.0]))
        np.testing.assert_allclose(prec[2], np.asarray([1.0]))


class MockDataset:
    def __init__(self):
        self.num_classes = 2

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
        output = PanopticResults(
            semantic_logits=torch.tensor([[0, 1], [0, 1], [1, 0]]).float(),
            offset_logits=None,
            cluster_scores=torch.tensor([0.5, 1]),
            clusters=[torch.tensor([0]), torch.tensor([1, 2])],
            cluster_type=None,
        )
        labels = PanopticLabels(
            center_label=None,
            y=torch.tensor([1, 1, 1]),
            num_instances=torch.tensor([2]),
            instance_labels=torch.tensor([1, 1, 2]),
            instance_mask=None,
            vote_label=None,
        )
        self.outputs = [output]
        self.labels = [labels]

    def get_output(self):
        return self.outputs[self.iter]

    def get_labels(self):
        return self.labels[self.iter]

    def get_current_losses(self):
        return self.losses[self.iter]

    @property
    def device(self):
        return "cpu"


class TestPanopticTracker(unittest.TestCase):
    def test_track_basic(self):
        tracker = PanopticTracker(MockDataset())
        model = MockModel()
        tracker.track(
            model,
            data=Data(pos=torch.tensor([[1, 2]]), batch=torch.tensor([0, 0, 0])),
            min_cluster_points=0,
            iou_threshold=0.25,
        )
        metrics = tracker.get_metrics()
        self.assertAlmostEqual(metrics["train_Iacc"], 1)
        self.assertAlmostEqual(metrics["train_pos"], 1)
        self.assertAlmostEqual(metrics["train_neg"], 0)

    def test_track_finalise(self):
        tracker = PanopticTracker(MockDataset())
        model = MockModel()
        tracker.track(
            model,
            data=Data(pos=torch.tensor([[1, 2]]), batch=torch.tensor([0, 0, 0])),
            min_cluster_points=0,
            iou_threshold=0.25,
            track_instances=True,
        )
        tracker.finalise(
            track_instances=True, iou_threshold=0.25,
        )
        metrics = tracker.get_metrics()
        self.assertAlmostEqual(metrics["train_map"], 1)


if __name__ == "__main__":
    unittest.main()
