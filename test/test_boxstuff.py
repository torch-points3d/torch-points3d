import os
import sys
import unittest
import torch
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points3d.utils.box_utils import (
    box_corners_from_param,
    box3d_vol,
    intersection_area,
    convex_hull_graham,
    nms_samecls,
    box3d_iou,
)
from torch_points3d.modules.VoteNet.votenet_results import VoteNetResults
from torch_points3d.metrics.box_detection.ap import eval_detection
from torch_points3d.datasets.object_detection.box_data import BoxData


class TestUtils(unittest.TestCase):
    def test_cornerfromparams(self):
        box = box_corners_from_param(torch.tensor([1, 2, 3]).float(), np.pi / 2, torch.tensor([1, 1, 1]))
        torch.testing.assert_allclose(
            box,
            1
            + torch.tensor(
                [
                    [1.0, -0.5, -1.5],
                    [1.0, 0.5, -1.5],
                    [-1.0, 0.5, -1.5],
                    [-1.0, -0.5, -1.5],
                    [1.0, -0.5, 1.5],
                    [1.0, 0.5, 1.5],
                    [-1.0, 0.5, 1.5],
                    [-1.0, -0.5, 1.5],
                ]
            ),
        )

    def test_box3dvol(self):
        box = box_corners_from_param(torch.tensor([1, 2, 3]).float(), np.pi / 2, torch.tensor([0, 0, 0]))
        self.assertEqual(box3d_vol(box), 6)

    def test_intersection_area(self):
        box1 = box_corners_from_param(torch.tensor([1, 1, 3]).float(), 0, torch.tensor([0, 0, 0])).numpy()
        box2 = box_corners_from_param(torch.tensor([1, 1, 3]).float(), np.pi / 2, torch.tensor([0, 0, 0])).numpy()
        rect1 = [(box1[i, 0], box1[i, 1]) for i in range(4)]
        rect2 = [(box2[i, 0], box2[i, 1]) for i in range(4)]
        self.assertAlmostEqual(intersection_area(rect1, rect2), 1, places=5)

        box1 = box_corners_from_param(torch.tensor([2, 2, 3]).float(), 0, torch.tensor([1, 1, 0])).numpy()
        box2 = box_corners_from_param(torch.tensor([2, 2, 3]).float(), 0, torch.tensor([0, 0, 0])).numpy()
        rect1 = [(box1[i, 0], box1[i, 1]) for i in range(4)]
        rect2 = [(box2[i, 0], box2[i, 1]) for i in range(4)]
        self.assertAlmostEqual(intersection_area(rect1, rect2), 1, places=5)

        rect1 = [(0, 0), (1, 0), (1, 1), (0, 1)]
        rect2 = [(0, 0), (1, 1), (0, 2), (-1, 1)]
        self.assertAlmostEqual(intersection_area(rect1, rect2), 0.5, places=5)

    def test_convexhull(self):
        rect = [(0, 0), (1, 0), (0.5, 0.5), (1, 1), (0, 1)]
        self.assertEqual(convex_hull_graham(rect), [(0, 0), (1, 0), (1, 1), (0, 1)])

    def test_nms(self):
        boxes = np.asarray([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0.5, 0.5, 0.5]])
        classes = np.asarray([0, 0])
        scores = np.asarray([0, 1])
        self.assertEqual(nms_samecls(boxes, classes, scores), [1, 0])
        self.assertEqual(nms_samecls(boxes, classes, scores, 0.1), [1])

    def test_box3diou(self):
        box1 = box_corners_from_param(torch.tensor([2, 2, 3]).float(), 0, torch.tensor([1, 1, 0])).numpy()
        box2 = box_corners_from_param(torch.tensor([1, 1, 1]).float(), 0, torch.tensor([0.5, 0.5, 0.5])).numpy()
        self.assertAlmostEqual(box3d_iou(box1, box2), 1.0 / (2 * 3 * 2), places=5)


class TestVotenetResults(unittest.TestCase):
    def test_nms(self):
        res = VoteNetResults(center=torch.zeros((2, 4, 3)))

        box = box_corners_from_param(torch.tensor([1, 1, 1]).float(), 0, torch.tensor([0.5, 0.5, 0.5]))
        boxes = box.unsqueeze(0).unsqueeze(0)
        boxes.repeat((res.batch_size, res.num_proposal, 1, 1))

        objectness = torch.tensor([[0, 1, 0.5, 0], [1, 0.8, 0, 0]])
        classes = torch.tensor([[0, 0, 0, 0], [2, 1, 1, 1]]).long()

        mask = res._nms_mask(boxes, objectness, classes)
        np.testing.assert_equal(mask, np.asarray([[False, True, False, False], [True, True, False, False]]))

    def test_getboxes(self):
        class MockDataset:
            def class2size(self, classname, residual):
                sizes = torch.tensor([[1, 1, 1], [0, 0, 0]]).float()
                return sizes[classname] + residual

            def class2angle(self, classname, residual):
                return 0

        batch_size = 2
        num_samples = 4
        results = VoteNetResults(
            center=torch.zeros((2, 4, 3)),
            heading_scores=torch.zeros(batch_size, num_samples, 1),
            heading_residuals=torch.zeros(batch_size, num_samples, 1),
            size_scores=torch.zeros(batch_size, num_samples, 1),
            size_residuals=torch.zeros(batch_size, num_samples, 2, 3),
            objectness_scores=torch.ones(batch_size, num_samples, 2),
            sem_cls_scores=torch.ones(batch_size, num_samples, 1),
        )

        boxes = results.get_boxes(MockDataset())
        self.assertEqual(len(boxes), batch_size)
        self.assertEqual(len(boxes[0]), num_samples)
        box = boxes[0][0]

        np.testing.assert_allclose(
            box.corners3d,
            np.asarray(
                [
                    [-0.5, -0.5, -0.5],
                    [0.5, -0.5, -0.5],
                    [0.5, 0.5, -0.5],
                    [-0.5, 0.5, -0.5],
                    [-0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                ]
            ),
        )
        self.assertAlmostEqual(box.score, 0.5)


class TestAP(unittest.TestCase):
    def test_evaldetection(self):
        box = box_corners_from_param(torch.tensor([1, 1, 1]).float(), 0, torch.tensor([0.5, 0.5, 0.5]))

        # Image1 -> 1 class1 and 1 class2
        # Image2 -> 1 class1
        gt = {
            "0": [BoxData("class1", box), BoxData("class2", box)],
            "1": [BoxData("class1", box)],
        }

        pred = {
            "0": [BoxData("class1", box, score=0.5), BoxData("class2", box, score=0.5)],
            "1": [BoxData("class2", box, score=1)],
        }
        rec, prec, ap = eval_detection(pred, gt)
        np.testing.assert_allclose(rec["class2"], np.asarray([0, 1]))
        np.testing.assert_allclose(prec["class2"], np.asarray([0, 0.5]))
        self.assertAlmostEqual(ap["class2"], 0.5)


if __name__ == "__main__":
    unittest.main()
