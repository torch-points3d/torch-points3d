import os
import sys
import numpy as np
import unittest
import tempfile
import h5py
import numpy.testing as npt
import numpy.matlib

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, ".."))

from torch_points3d.metrics.confusion_matrix import ConfusionMatrix


class TestConfusionMatrix(unittest.TestCase):
    def setUp(self):
        matrix = np.asarray([[4, 1], [2, 10]])
        self._confusion = ConfusionMatrix.create_from_matrix(matrix)

    def test_getCount(self):
        self.assertEqual(self._confusion.get_count(0, 0), 4)

    def test_getIoU(self):
        iou = self._confusion.get_intersection_union_per_class()[0]
        self.assertAlmostEqual(iou[0], 4 / (4.0 + 1.0 + 2.0))
        self.assertAlmostEqual(iou[1], 10 / (10.0 + 1.0 + 2.0))

    def test_getIoUMissing(self):
        matrix = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        confusion = ConfusionMatrix.create_from_matrix(matrix)
        iou, mask = confusion.get_intersection_union_per_class()
        self.assertAlmostEqual(iou[0], 1)
        self.assertAlmostEqual(iou[1], 1)
        self.assertAlmostEqual(iou[2], 0)
        npt.assert_array_equal(mask, np.array([True, True, False]))

    def test_getMeanIoU(self):
        iou = self._confusion.get_intersection_union_per_class()[0]
        self.assertAlmostEqual(iou[0], 4 / (4.0 + 1.0 + 2.0))
        self.assertAlmostEqual(iou[1], 10 / (10.0 + 1.0 + 2.0))

    def test_test_getMeanIoUMissing(self):
        matrix = np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 0]])
        confusion = ConfusionMatrix.create_from_matrix(matrix)
        self.assertAlmostEqual(confusion.get_average_intersection_union(missing_as_one=False), (0.5 + 0.5) / 2)
        self.assertAlmostEqual(confusion.get_average_intersection_union(missing_as_one=True), (0.5 + 1 + 0.5) / 3)


if __name__ == "__main__":
    unittest.main()
