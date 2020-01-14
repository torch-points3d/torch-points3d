import os
import sys
import numpy as np
import torch
import unittest
import tempfile
import h5py
import numpy.testing as npt
import numpy.matlib
from torch_geometric.data import Data

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, ".."))

import datasets.transforms as cT


class TestGridSampling(unittest.TestCase):
    def setUp(self):

        num_classes = 2
        self.sampler = cT.GridSampling(0.04, num_classes=num_classes)

        num_points = 5

        pos = torch.from_numpy(np.array([[0, 0, 0.01], [0.01, 0, 0], [0, 0.01, 0], [0, 0.01, 0], [0.01, 0, 0.01]]))

        batch = torch.from_numpy(np.zeros(num_points))

        y = np.asarray(np.random.randint(0, 1, 5))
        self._uniq, _ = np.unique(y, return_counts=True)

        y = torch.from_numpy(y)

        self.data = Data(pos=pos, batch=batch, y=y)

    def test_simple(self):

        out = self.sampler(self.data)

        y = out.y.detach().cpu().numpy()

        npt.assert_array_almost_equal(self._uniq, y)


if __name__ == "__main__":
    unittest.main()
