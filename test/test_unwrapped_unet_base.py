import unittest
import importlib
from omegaconf import OmegaConf
import torch
from torch_geometric.data import Data, Batch
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)

from src.utils.model_building_utils.model_definition_resolver import resolve_model
from src.models.base_architectures import UnwrappedUnetBasedModel
from src.utils.config import merges_in_sub, set_format

from test.mockdatasets import MockDataset


class SegmentationModel(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnwrappedUnetBasedModel
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)
        pass


class TestModelDefinitionResolver(unittest.TestCase):
    def test_resolve_1(self):
        models_conf = os.path.join(ROOT, "test/config_unwrapped_unet_base/test_models.yaml")
        config = os.path.join(ROOT, "conf/config.yaml")

        models_conf = OmegaConf.load(models_conf).models

        config = OmegaConf.load(config)

        cfg_training = config.training
        cfg_dataset = config.data.s3dis

        dataset = MockDataset(6)
        tested_task = "segmentation"

        resolve_model(models_conf, dataset, tested_task)

        for _, model_conf in models_conf.items():
            model_class = model_conf.architecture
            model_paths = model_class.split(".")
            module = ".".join(model_paths[:-1])
            model_module = ".".join(["src.models", tested_task, module])
            modellib = importlib.import_module(model_module)
            model = SegmentationModel(model_conf, "", dataset, modellib)

            assert len(model.down_modules) == len(model_conf.down_conv.down_conv_nn)
            assert len(model.up_modules) == len(model_conf.up_conv.up_conv_nn)

            # innermost is not tested yet


if __name__ == "__main__":
    unittest.main()
