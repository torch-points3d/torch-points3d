import os
import sys
import unittest
import numpy as np
import omegaconf
from omegaconf import OmegaConf

import torch
from torch_geometric.data import Data

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ROOT)

from torch_points3d.datasets.segmentation.shapenet import ShapeNetDataset
from torch_points3d.datasets.dataset_factory import get_dataset_class

def load_dataconfig(task, dataset):
    data_conf = os.path.join(ROOT, "..", "conf/data/{}/{}.yaml".format(task, dataset))
    config = OmegaConf.load(data_conf)
    if omegaconf.__version__ == '1.4.1':
        config.update("data.task", "segmentation")
        config.update("data.dataroot", "data")
    else:
        OmegaConf.update(config, "data.task", "segmentation", merge=True)
        OmegaConf.update(config, "data.dataroot", "data", merge=True)

    return config

class TestBaseFactory(unittest.TestCase):
    def test_simple(self):
        data_config = load_dataconfig("segmentation", "shapenet")
        dataset_cls = get_dataset_class(data_config)
        self.assertEqual("<class 'torch_points3d.datasets.segmentation.shapenet.ShapeNetDataset'>", str(dataset_cls))

if __name__ == "__main__":
    unittest.main()
