import unittest
from omegaconf import OmegaConf
import os
import sys
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from glob import glob

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from test.mockdatasets import MockDatasetGeometric, MockDataset


from src import find_model_using_name
from src.utils.model_building_utils.model_definition_resolver import resolve_model
from src.utils.config import set_format, merges_in_sub

# calls resolve_model, then find_model_using_name


def _find_model_using_name(model_class, task, model_config, dataset):
    resolve_model(model_config, dataset, task)
    return find_model_using_name(model_class, task, model_config, dataset)


def load_model_config(task, model_type):
    models_conf = os.path.join(ROOT, "conf/models/{}/{}.yaml".format(task, model_type))
    config = OmegaConf.load(models_conf)
    return config.models


class TestModelUtils(unittest.TestCase):
    def setUp(self):
        self.model_type_files = glob(os.path.join(ROOT, "conf/models/*/*.yaml"))

    def test_createall(self):
        for type_file in self.model_type_files:

            associated_task = type_file.split("/")[-2]
            models_config = OmegaConf.load(type_file).models
            for model_name in models_config.keys():
                print(model_name)
                if model_name not in ["MyTemplateModel"]:
                    model_config = models_config[model_name]
                    _find_model_using_name(
                        model_config.architecture, associated_task, model_config, MockDatasetGeometric(6)
                    )

    def test_pointnet2(self):
        params = load_model_config("segmentation", "pointnet2")["pointnet2"]
        dataset = MockDatasetGeometric(5)
        model = _find_model_using_name(params.architecture, "segmentation", params, dataset)
        model.set_input(dataset[0])
        model.forward()
        model.backward()

    def test_kpconv(self):
        params = load_model_config("segmentation", "kpconv")["SimpleKPConv"]
        dataset = MockDatasetGeometric(5)
        model = _find_model_using_name(params.architecture, "segmentation", params, dataset)
        model.set_input(dataset[0])
        model.forward()
        model.backward()

    def test_pointnet2ms(self):
        params = load_model_config("segmentation", "pointnet2")["pointnet2ms"]
        dataset = MockDatasetGeometric(5)
        model = _find_model_using_name(params.architecture, "segmentation", params, dataset)
        model.set_input(dataset[0])
        model.forward()
        model.backward()

    # def test_pointnet2_customekernel(self):
    #     model_type = 'pointnet2_dense'
    #     params = self.config['models']['pointnet2_kc']
    #     dataset = MockDataset(5)
    #     model = _find_model_using_name(model_type, 'segmentation', params, dataset)
    #     model.set_input(dataset[0])
    #     model.forward()


if __name__ == "__main__":
    unittest.main()
