import unittest

from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)

from torch_points3d.utils.model_building_utils.model_definition_resolver import resolve_model


class MockDataset(Dataset):
    def __init__(self, feature_size=6):
        self.feature_dimension = feature_size
        self.num_classes = 10
        self.weight_classes = None
        nb_points = 100
        self._pos = torch.randn((nb_points, 3))
        if feature_size > 0:
            self._feature = torch.tensor([range(feature_size) for i in range(self._pos.shape[0])], dtype=torch.float,)
        else:
            self._feature = None
        self._y = torch.tensor([range(10) for i in range(self._pos.shape[0])], dtype=torch.float)
        self._batch = torch.tensor([0 for i in range(self._pos.shape[0])])

    def __getitem__(self, index):
        return Batch(pos=self._pos, x=self._feature, y=self._y, batch=self._batch)


class TestModelDefinitionResolver(unittest.TestCase):
    def test_resolve_1(self):
        model_conf = os.path.join(ROOT, "test/test_config/test_resolver_in.yaml")
        config = OmegaConf.load(model_conf)
        dataset = MockDataset(6)
        tested_task = "segmentation"

        resolve_model(config, dataset, tested_task)

        expected = OmegaConf.load(os.path.join(ROOT, "test/test_config/test_resolver_out.yaml"))

        self.assertEqual(dict(config), dict(expected))


if __name__ == "__main__":
    unittest.main()
