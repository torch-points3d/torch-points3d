import os
import sys
import numpy as np
import unittest
import torch

import numpy.testing as npt
import numpy.matlib

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, ".."))

from torch_points3d.metrics.confusion_matrix import compute_average_intersection_union
from torch_points3d.metrics.confusion_matrix import compute_mean_class_accuracy
from torch_points3d.metrics.confusion_matrix import compute_overall_accuracy
from torch_points3d.metrics.confusion_matrix import compute_intersection_union_per_class


class TestConfusionMatrix(unittest.TestCase):

    def test_compute_intersection_union_per_class(self):
        matrix = torch.tensor([[4, 1], [2, 10]])
        iou, _ = compute_intersection_union_per_class(matrix)
        miou = compute_average_intersection_union(matrix)
        self.assertAlmostEqual(iou[0].item(), 4 / (4.0 + 1.0 + 2.0))
        self.assertAlmostEqual(iou[1].item(), 10 / (10.0 + 1.0 + 2.0))
        self.assertAlmostEqual(iou.mean().item(), miou.item())

    def test_compute_overall_accuracy(self):
        matrix = torch.tensor([[4, 1], [2, 10]]).float()
        acc = compute_overall_accuracy(matrix)
        self.assertAlmostEqual(acc.item(), (4.0+10.0)/(4.0 + 10.0 + 1.0 +2.0))

        # try int confusion matrix
        matrix = torch.tensor([[4, 1], [2, 10]]).int()
        acc = compute_overall_accuracy(matrix)
        self.assertAlmostEqual(acc.item(), (4.0+10.0)/(4.0 + 10.0 + 1.0 +2.0))

        matrix = torch.tensor([[0, 0], [0, 0]]).float()
        acc = compute_overall_accuracy(matrix)
        self.assertAlmostEqual(acc.item(), 0.0)

    def test_compute_mean_class_accuracy(self):
        matrix = torch.tensor([[4, 1], [2, 10]]).float()
        macc = compute_mean_class_accuracy(matrix)
        self.assertAlmostEqual(macc.item(), (4/5 + 10/12)*0.5)



    def test_test_getMeanIoUMissing(self):
        matrix = torch.tensor([[1, 1, 0], [0, 1, 0], [0, 0, 0]])
        self.assertAlmostEqual(compute_average_intersection_union(matrix, missing_as_one=False), (0.5 + 0.5) / 2)
        self.assertAlmostEqual(compute_average_intersection_union(matrix, missing_as_one=True), (0.5 + 1 + 0.5) / 3)




if __name__ == "__main__":
    unittest.main()
