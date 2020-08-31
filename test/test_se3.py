import unittest
import sys
import os

import torch


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, ".."))
torch.manual_seed(0)

from torch_points3d.core.geometry.se3 import SE3Transform


class Testhelpers(unittest.TestCase):
    def test_batch_transform(self):
        pass

    def test_multi_batch_transform(self):
        pass

    def test_partial_transform(self):
        pass

    def test_multi_partial_transform(self):
        pass
