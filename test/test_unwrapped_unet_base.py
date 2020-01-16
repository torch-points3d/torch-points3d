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
from models.unwrapped_unet_base import UnwrappedUnetBasedModel
from utils_folder.utils import merges_in_sub, set_format


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

    def set_input(self, data):
        pass

    def forward(self):
        r"""
            Forward pass of the network
            self.data:
                x -- Features [B, C, N]
                pos -- Features [B, N, 3]
        """
        stack_down = []
        queue_up = queue.Queue()

        data = self.input
        stack_down.append(data)

        for i in range(len(self.down_modules)):
            data = self.down_modules[i](data)
            stack_down.append(data)

        data_inner = self.inner_modules[0](data)
        queue_up.put(data_inner)

        for i in range(len(self.up_modules)):
            data = self.up_modules[i]((queue_up.get(), stack_down.pop()))
            queue_up.put(data)

        last_feature = torch.cat([data.x, data_inner.x.repeat(1, 1, data.x.shape[-1])], dim=1)

        if self._use_category:
            num_points = data.pos.shape[1]
            cat_one_hot = (
                torch.zeros((data.pos.shape[0], self._num_categories, num_points)).float().to(self.category.device)
            )
            cat_one_hot.scatter_(1, self.category.repeat(1, num_points).unsqueeze(1), 1)
            last_feature = torch.cat((last_feature, cat_one_hot), dim=1)

        self.output = self.FC_layer(last_feature).transpose(1, 2).contiguous().view((-1, self._num_classes))
        return self.output

    def backward(self):
        pass


class MockDataset(torch.utils.data.Dataset):
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


if __name__ == "__main__":
    unittest.main()
