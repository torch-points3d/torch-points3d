import unittest
from omegaconf import OmegaConf
import os
import sys
from torch_geometric.data import Data
import torch

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(ROOT)

from datasets.base_dataset import BaseDataset


class MockDataset(torch.utils.data.Dataset):
    def __init__(self, feature_size):
        if feature_size > 0:
            self._feature = torch.tensor([range(feature_size)])
        else:
            self._feature = None

    def __getitem__(self, index):
        return Data(pos=torch.tensor([[0, 0, 0]]), x=self._feature)


class TestDataset(unittest.TestCase):
    def test_extractdims(self):
        dataset = MockDataset(2)
        self.assertEqual(BaseDataset.extract_point_dimension(dataset), 2)

        dataset = MockDataset(0)
        self.assertEqual(BaseDataset.extract_point_dimension(dataset), 0)


if __name__ == "__main__":
    unittest.main()
