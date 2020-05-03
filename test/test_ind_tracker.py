import os
import sys
import numpy as np
import torch
import unittest
import h5py
import numpy.testing as npt
import numpy.matlib
from torch_geometric.data import Data

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, ".."))

from torch_points3d.datasets.registration.utils import tracked_matches


class TestTrackedMatches(unittest.TestCase):
    def test_simple(self):

        ind_source = torch.tensor([1, 2, 5])
        ind_target = torch.tensor([0, 5, 6])
        data_s = Data(pos=torch.randn(3, 3), origin_id=ind_source)
        data_t = Data(pos=torch.randn(3, 3), origin_id=ind_target)
        pair = torch.tensor([[0, 2], [1, 3], [2, 0], [3, 1]])

        res = tracked_matches(data_s, data_t, pair)
        expected = np.array([[1, 0]])
        npt.assert_array_almost_equal(res.detach().cpu().numpy(), expected)
