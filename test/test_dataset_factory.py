import os
import sys
import unittest
import numpy as np
from omegaconf import OmegaConf

import torch
from torch_geometric.data import Data

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ROOT)

from src.datasets.segmentation.shapenet import ShapeNetDataset
from src.datasets.dataset_factory import get_dataset_class

def load_dataconfig(task, dataset):
    data_conf = os.path.join(ROOT, "..", "conf/data/{}/{}.yaml".format(task, dataset))
    config = OmegaConf.load(data_conf)
    config.update("data.task", "segmentation")
    config.update("data.dataroot", "data")
    return config

class TestBaseFactory(unittest.TestCase):
    def test_simple(self):
        data_config = load_dataconfig("segmentation", "shapenet")
        dataset_cls = get_dataset_class(data_config.data)
        self.assertEqual("<class 'src.datasets.segmentation.shapenet.ShapeNetDataset'>", str(dataset_cls))

if __name__ == "__main__":
    unittest.main()
