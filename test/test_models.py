import unittest
from omegaconf import OmegaConf
import os
import sys
import torch
from torch_geometric.data import Data, Batch

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)

from models.utils import find_model_using_name
from models.pointnet2.nn import SegmentationModel
from models.model_building_utils.model_definition_resolver import resolve_model
from test.mockdatasets import MockDatasetGeometric, MockDataset

# calls resolve_model, then find_model_using_name


def _find_model_using_name(model_type, task, model_config, dataset):
    resolve_model(model_config, dataset, task)
    return find_model_using_name(model_type, task, model_config, dataset)


class TestModelUtils(unittest.TestCase):
    def setUp(self):
        models_conf = os.path.join(ROOT, "conf/models/segmentation.yaml")
        config_file = OmegaConf.load(os.path.join(ROOT, "conf/config.yaml"))

        self.config = OmegaConf.load(models_conf)
        self.config = OmegaConf.merge(self.config, config_file.training)

    def test_createall(self):
        for model_name in self.config["models"].keys():
            print(model_name)
            if model_name not in ["MyTemplateModel", "Randlanet_Res", "Randlanet_Conv"]:
                params = self.config["models"][model_name]
                _find_model_using_name(params.type, "segmentation", params, MockDatasetGeometric(6))

    def test_pointnet2(self):
        model_type = "pointnet2"
        params = self.config["models"][model_type]
        dataset = MockDatasetGeometric(5)
        model = _find_model_using_name(model_type, "segmentation", params, dataset)
        model.set_input(dataset[0])
        model.forward()

    def test_kpconv(self):
        model_type = "KPConv"
        params = self.config["models"]["SimpleKPConv"]
        dataset = MockDatasetGeometric(5)
        model = _find_model_using_name(model_type, "segmentation", params, dataset)
        model.set_input(dataset[0])
        model.forward()

    def test_pointnet2ms(self):
        model_type = "pointnet2"
        params = self.config["models"]["pointnet2ms"]
        dataset = MockDatasetGeometric(5)
        model = _find_model_using_name(model_type, "segmentation", params, dataset)
        model.set_input(dataset[0])
        model.forward()

    # def test_pointnet2_customekernel(self):
    #     model_type = 'pointnet2_dense'
    #     params = self.config['models']['pointnet2_kc']
    #     dataset = MockDataset(5)
    #     model = _find_model_using_name(model_type, 'segmentation', params, dataset)
    #     model.set_input(dataset[0])
    #     model.forward()


if __name__ == "__main__":
    unittest.main()
