import unittest
import importlib
from omegaconf import OmegaConf
import torch
from torch_geometric.data import Data, Batch
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)

from models.model_building_utils.model_definition_resolver import resolve_model
from models.unet_base import UnwrappedUnetBasedModel
from utils_folder.utils import merges_in_sub, set_format

from test.mockdatasets import MockDataset


class SegmentationModel(UnwrappedUnetBasedModel):
    r"""
        RSConv Segmentation Model with / without multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

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

        for model_name, model_conf in models_conf.items():

            print(model_name)
            model_type = model_conf.type
            module_filename = ".".join(["models", model_type, "modules"])
            modules_lib = importlib.import_module(module_filename)

            cfg_training = set_format(model_conf, cfg_training)
            model_conf = merges_in_sub(model_conf, [cfg_training, cfg_dataset])

            model = SegmentationModel(model_conf, model_type, dataset, modules_lib)

            assert len(model.down_modules) == len(model_conf.down_conv.down_conv_nn)
            assert len(model.up_modules) == len(model_conf.up_conv.up_conv_nn)

            # innermost is not tested yet


if __name__ == "__main__":
    unittest.main()
