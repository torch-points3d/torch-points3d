import unittest
from models.utils import find_model_using_name
from omegaconf import OmegaConf
import os

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')


class TestModelUtils(unittest.TestCase):
    def setUp(self):
        config_file = os.path.join(ROOT, 'conf/models/segmentation.yaml')
        self.config = OmegaConf.load(config_file)

    def test_findmodel(self):
        for model_name in ['KPConv', 'RSConv', 'pointnet2']:
            params = self.config['models'][model_name]
            model = find_model_using_name(model_name, 'segmentation', params, 10)


if __name__ == "__main__":
    unittest.main()
