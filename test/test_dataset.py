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

    def __getitem__(self, index):
        return Data(pos=torch.tensor([[0, 0, 0]]), x=torch.tensor([[0, 0, 0]]))


class TestDataset(unittest.TestCase):
    def test_extractdims(self):
        dataset = MockDataset()
        self.assertEqual(BaseDataset.extract_point_dimension(dataset), 6)


if __name__ == "__main__":
    unittest.main()
