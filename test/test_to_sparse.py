import unittest
import sys
import os
import numpy as np
import torch
from torch_geometric.data import Data
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, ".."))

from src.core.data_transform import ToSparseInput
from src.utils.enums import ConvolutionFormat
from src.datasets.multiscale_data import MultiScaleBatch


class TestSparse(unittest.TestCase):
    def test_to_sparse_input(self):
        arr = np.asarray([[0, 0, 0], [0, 1, 0], [0, 1, 0.25], [0.25, 0.25, 0]])
        feat = torch.tensor([0, 1, 2, 3])
        data = Data(pos=torch.from_numpy(arr), x=feat)
        transform = ToSparseInput(grid_size=1, save_delta=True, save_delta_norm=True, mode="last")
        data_out = transform(data.clone())

        self.assertIn("delta_norm", data_out.keys)
        self.assertIn("delta", data_out.keys)
        self.assertEqual(data_out.pos.dtype, torch.int)
        self.assertEqual(2, data_out.pos.shape[0])
        torch.testing.assert_allclose(data_out.x, torch.tensor([3, 2]))

    def test_to_sparse_input_mean(self):
        arr = np.asarray([[0, 0, 0], [0, 1, 0], [0, 1, 0.25], [0.25, 0.25, 0]])
        feat = torch.tensor([0, 1.0, 2.0, 4.0])
        data = Data(pos=torch.from_numpy(arr), x=feat)
        transform = ToSparseInput(grid_size=1, save_delta=True, save_delta_norm=True, mode="mean")
        data_out = transform(data.clone())

        self.assertIn("delta_norm", data_out.keys)
        self.assertIn("delta", data_out.keys)
        self.assertEqual(data_out.pos.dtype, torch.int)
        self.assertEqual(2, data_out.pos.shape[0])
        torch.testing.assert_allclose(data_out.x, torch.tensor([2, 1.5]))


if __name__ == "__main__":
    unittest.main()
