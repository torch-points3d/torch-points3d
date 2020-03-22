import unittest
import torch
from torch_geometric.data import Data

import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)

from src.datasets.base_dataset import BaseDataset

class TestBaseDataset(unittest.TestCase):
    def test_base_dataset(self):

        dataset = BaseDataset()

if __name__ == "__main__":
    unittest.main()
