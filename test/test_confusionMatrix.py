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

from metrics.confusion_matrix import ConfusionMatrix


class TestConfusionMatrix(unittest.TestCase):
    def setUp(self):
        matrix = np.asarray([[4, 1], [2, 10]])
        conf = ConfusionMatrix(2)
        for i in range(2):
            for j in range(2):
                conf.count_predicted(i, j, matrix[i][j])
        self._confusion = conf

    def test_getCount(self):
        self.assertEqual(self._confusion.get_count(0, 0), 4)

    def test_getIoU(self):
        iou = self._confusion.get_intersection_union_per_class()
        self.assertAlmostEqual(iou[0], 4 / (4.0 + 1.0 + 2.0))
        self.assertAlmostEqual(iou[1], 10 / (10.0 + 1.0 + 2.0))

    def test_getMeanIoU(self):
        iou = self._confusion.get_intersection_union_per_class()
        self.assertAlmostEqual(iou[0], 4 / (4.0 + 1.0 + 2.0))
        self.assertAlmostEqual(iou[1], 10 / (10.0 + 1.0 + 2.0))


if __name__ == "__main__":
    unittest.main()
