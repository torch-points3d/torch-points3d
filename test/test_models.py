import unittest
from omegaconf import OmegaConf
import os
import sys
import numpy as np
import torch
from torch_geometric.data import Data, Batch

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


def _find_random_dataset(datasets):
    dataset_names = datasets.keys()
    idx = np.random.choice(len(dataset_names))
    return getattr(datasets, list(dataset_names)[idx])


def load_model_config(task, model_type, training):
    models_conf = os.path.join(ROOT, "conf/models/{}/{}.yaml".format(task, model_type))
    config = OmegaConf.load(models_conf)
    config = OmegaConf.merge(config, training)
    return getattr(config.models, task)

class TestModelUtils(unittest.TestCase):
    def setUp(self):
        self.cfg_training = OmegaConf.load(os.path.join(ROOT, "conf/training.yaml"))
    """
    def test_createall(self):
        for model_name in self.config["models"].keys():
            print(model_name)
            if model_name not in ["MyTemplateModel"]:
                model_config = self.config["models"][model_name]

                cfg_training = set_format(model_config, self.config_file.training)
                model_config = merges_in_sub(model_config, [cfg_training, _find_random_dataset(self.config_file.data)])

                _find_model_using_name(model_config.architecture, "segmentation", model_config, MockDatasetGeometric(6))
    """
    
    def test_pointnet2(self):
        params = load_model_config("segmentation", "pointnet2", self.cfg_training)['pointnet2']
        dataset = MockDatasetGeometric(5)
        model = _find_model_using_name(params.architecture, "segmentation", params, dataset)
        model.set_input(dataset[0])
        model.forward()

    def test_kpconv(self):
        params = load_model_config("segmentation", "kpconv", self.cfg_training)['SimpleKPConv']
        dataset = MockDatasetGeometric(5)
        model = _find_model_using_name(params.architecture, "segmentation", params, dataset)
        model.set_input(dataset[0])
        model.forward()

    def test_pointnet2ms(self):
        params = load_model_config("segmentation", "pointnet2", self.cfg_training)['pointnet2ms']
        dataset = MockDatasetGeometric(5)
        model = _find_model_using_name(params.architecture, "segmentation", params, dataset)
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
