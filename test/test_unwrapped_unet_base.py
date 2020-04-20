import unittest
from omegaconf import OmegaConf
import torch
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)

from torch_points3d.utils.model_building_utils.model_definition_resolver import resolve_model
from torch_points3d.models.base_architectures import UnwrappedUnetBasedModel

from test.mockdatasets import MockDataset


class MockModel(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnwrappedUnetBasedModel
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)


class ConvMock(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        pass


class MockModelLib:
    ConvMock = ConvMock


class TestModelDefinitionResolver(unittest.TestCase):
    def test_resolve_1(self):
        models_conf = os.path.join(ROOT, "test/config_unwrapped_unet_base/test_models.yaml")
        models_conf = OmegaConf.load(models_conf).models

        dataset = MockDataset(6)
        tested_task = "segmentation"

        resolve_model(models_conf, dataset, tested_task)

        for _, model_conf in models_conf.items():
            modellib = MockModelLib()
            model = MockModel(model_conf, "", dataset, modellib)

            assert len(model.down_modules) == len(model_conf.down_conv.down_conv_nn)
            assert len(model.up_modules) == len(model_conf.up_conv.up_conv_nn)

            # innermost is not tested yet


if __name__ == "__main__":
    unittest.main()
