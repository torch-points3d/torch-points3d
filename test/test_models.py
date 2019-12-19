import unittest
from omegaconf import OmegaConf
import os
import sys
import torch
from torch_geometric.data import Data, Batch
ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(ROOT)

from models.utils import find_model_using_name
from models.pointnet2.nn import SegmentationModel


class MockDataset(torch.utils.data.Dataset):
    def __init__(self, feature_size=0):
        self.feature_dimension = feature_size
        self.num_classes = 10
        nb_points = 100
        self._pos = torch.randn((nb_points, 3))
        if feature_size > 0:
            self._feature = torch.tensor([range(feature_size) for i in range(self._pos.shape[0])], dtype=torch.float)
        else:
            self._feature = None
        self._y = torch.tensor([range(10) for i in range(self._pos.shape[0])], dtype=torch.float)
        self._batch = torch.tensor([0 for i in range(self._pos.shape[0])])

    def __getitem__(self, index):
        return Batch(pos=self._pos, x=self._feature,
                     y=self._y, batch=self._batch)


class TestModelUtils(unittest.TestCase):
    def setUp(self):
        models_conf = os.path.join(ROOT, 'conf/models/segmentation.yaml')
        config_file = OmegaConf.load(os.path.join(ROOT, 'conf/config.yaml'))

        self.config = OmegaConf.load(models_conf)
        self.config = OmegaConf.merge(self.config, config_file.training)

    def test_createall(self):
        for model_name in self.config['models'].keys():
            print(model_name)
            if model_name not in ["MyTemplateModel"]:
                params = self.config['models'][model_name]
                find_model_using_name(params.type, 'segmentation', params, MockDataset())

    def test_pointnet2(self):
        model_type = 'pointnet2'
        params = self.config['models'][model_type]
        dataset = MockDataset(5)
        model = find_model_using_name(model_type, 'segmentation', params, dataset)
        model.set_input(dataset[0])
        model.forward()

    def test_kpconv(self):
        model_type = 'KPConv'
        params = self.config['models']['SimpleKPConv']
        dataset = MockDataset(5)
        model = find_model_using_name(model_type, 'segmentation', params, dataset)
        model.set_input(dataset[0])
        model.forward()


if __name__ == "__main__":
    unittest.main()
