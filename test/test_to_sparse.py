import unittest
import sys
import os
import numpy as np
import torch
from itertools import combinations
from torch_geometric.data import Data
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, ".."))

from torch_points3d.core.data_transform import ToSparseInput
from torch_points3d.utils.enums import ConvolutionFormat
from torch_points3d.datasets.multiscale_data import MultiScaleBatch


class TestSparse(unittest.TestCase):
    def test_to_sparse_input(self):
        arr = np.asarray([[0, 0, 0], [0, 1, 0], [0, 1, 0.25], [0.25, 0.25, 0]])
        np_feat = [0, 1, 2, 3]
        feat = torch.tensor(np_feat)
        data = Data(pos=torch.from_numpy(arr), x=feat)
        transform = ToSparseInput(grid_size=1, mode="last")
        data_out = transform(data.clone())

        self.assertEqual(data_out.pos.dtype, torch.int)
        self.assertEqual(2, data_out.pos.shape[0])

        combi = list(combinations(np_feat, 2))
        tensors = [torch.tensor(t) for t in combi] + [torch.tensor(t[::-1]) for t in combi]

        is_in = False
        for t in tensors:
            if torch.eq(data_out.x, t).sum().item() == 2:
                is_in = True
        self.assertEqual(is_in, True)

    def test_to_sparse_input_mean(self):
        arr = np.asarray([[0, 0, 0], [0, 1, 0], [0, 1, 0.25], [0.25, 0.25, 0]])
        feat = torch.tensor([0, 1.0, 2.0, 4.0])
        data = Data(pos=torch.from_numpy(arr), x=feat)
        transform = ToSparseInput(grid_size=1, mode="mean")
        data_out = transform(data.clone())

        self.assertEqual(data_out.pos.dtype, torch.int)
        self.assertEqual(2, data_out.pos.shape[0])
        torch.testing.assert_allclose(data_out.x, torch.tensor([2, 1.5]))


if __name__ == "__main__":
    unittest.main()
