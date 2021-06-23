import os
import sys
import unittest
import numpy as np

import torch
from torch_geometric.data import Data

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ROOT)

from torch_points3d.datasets.segmentation.shapenet import ShapeNetDataset
from torch_points3d.datasets.dataset_factory import get_dataset_class
from test.utils import load_hydra_config


class TestBaseFactory(unittest.TestCase):
    def test_simple(self):
        data_config = load_hydra_config("data", "segmentation", "shapenet", "shapenet", {"++data.task=segmentation", "++data.dataroot=data"}).data

        dataset_cls = get_dataset_class(data_config)
        self.assertEqual("<class 'torch_points3d.datasets.segmentation.shapenet.ShapeNetDataset'>", str(dataset_cls))

if __name__ == "__main__":
    unittest.main()
