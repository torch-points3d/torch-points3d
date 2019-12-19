import unittest
from omegaconf import OmegaConf
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(ROOT)

from models.utils import find_model_using_name
from models.pointnet2.nn import SegmentationModel


class MockDataset:
    num_classes = 10
    feature_dimension = 5


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

    # def test_pointnet2(self):
    #     params = self.config['models']['pointnet2']
    #     SegmentationModel(params,)


if __name__ == "__main__":
    unittest.main()
