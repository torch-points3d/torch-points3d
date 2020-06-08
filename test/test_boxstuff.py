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
)


class TestUtils(unittest.TestCase):
    def test_cornerfromparams(self):
        box = box_corners_from_param(torch.tensor([1, 2, 3]), np.pi / 2, torch.tensor([1, 1, 1]))
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
        box = box_corners_from_param(torch.tensor([1, 2, 3]), np.pi / 2, torch.tensor([0, 0, 0]))
        self.assertEqual(box3d_vol(box), 6)

    def test_intersection_area(self):
        box1 = box_corners_from_param(torch.tensor([1, 1, 3]), 0, torch.tensor([0, 0, 0])).numpy()
        box2 = box_corners_from_param(torch.tensor([1, 1, 3]), np.pi / 2, torch.tensor([0, 0, 0])).numpy()
        rect1 = [(box1[i, 0], box1[i, 1]) for i in range(4)]
        rect2 = [(box2[i, 0], box2[i, 1]) for i in range(4)]
        self.assertAlmostEqual(intersection_area(rect1, rect2), 1, places=5)

        box1 = box_corners_from_param(torch.tensor([2, 2, 3]), 0, torch.tensor([1, 1, 0])).numpy()
        box2 = box_corners_from_param(torch.tensor([2, 2, 3]), 0, torch.tensor([0, 0, 0])).numpy()
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


if __name__ == "__main__":
    unittest.main()
