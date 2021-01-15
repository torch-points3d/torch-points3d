import unittest
from omegaconf import OmegaConf
import torch
import os
import sys
import copy

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)

from torch_points3d.utils.model_building_utils.model_definition_resolver import resolve_model
from torch_points3d.models.base_architectures import UnwrappedUnetBasedModel


class MockModel(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnwrappedUnetBasedModel
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)


class ConvMockDown(torch.nn.Module):
    def __init__(self, test_precompute=False, *args, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.test_precompute = test_precompute

    def forward(self, data, *args, **kwargs):
        data.append(self.kwargs["down_conv_nn"])
        if self.test_precompute:
            assert kwargs["precomputed"] is not None
        return data


class InnerMock(torch.nn.Module):
    def __init__(self, test_precompute=False, *args, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, data, *args, **kwargs):
        data.append("inner")
        return data


class ConvMockUp(torch.nn.Module):
    def __init__(self, test_precompute=False, *args, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.test_precompute = test_precompute

    def forward(self, data, *args, **kwargs):
        data = data[0].copy()
        data.append(self.kwargs["up_conv_nn"])
        if self.test_precompute:
            assert kwargs["precomputed"] is not None
        return data


class MockModelLib:
    ConvMockUp = ConvMockUp
    ConvMockDown = ConvMockDown
    InnerMock = InnerMock


class TestUnwrapperUnet(unittest.TestCase):
    def test_forward(self):
        models_conf = os.path.join(ROOT, "test/config_unwrapped_unet_base/test_models.yaml")
        models_conf = OmegaConf.load(models_conf).models
        models_conf_clone = copy.deepcopy(models_conf)
        modellib = MockModelLib()
        model = MockModel(models_conf["TestUnwrapper"], "", None, modellib)
        data = []
        d = model(data)
        self.assertEqual(d, [0, 1, 2, 3, "inner", 4, 5, 6, 7])
        self.assertEqual(models_conf_clone, models_conf)

    def test_forwardprecompute(self):
        models_conf = os.path.join(ROOT, "test/config_unwrapped_unet_base/test_models.yaml")
        models_conf = OmegaConf.load(models_conf).models
        modellib = MockModelLib()
        model = MockModel(models_conf["TestPrecompute"], "", None, modellib)
        data = []
        d = model(data, precomputed_up="Hey", precomputed_down="Yay")
        self.assertEqual(d, [0, 1, 2, 3, "inner", 4, 5, 6, 7])

    def test_noinnermost(self):
        models_conf = os.path.join(ROOT, "test/config_unwrapped_unet_base/test_models.yaml")
        models_conf = OmegaConf.load(models_conf).models
        modellib = MockModelLib()
        model = MockModel(models_conf["TestNoInnermost"], "", None, modellib)
        data = []
        d = model(data, precomputed_up="Hey", precomputed_down="Yay")
        self.assertEqual(d, [0, 1, 2, 3, 4, 5, 6])

    def test_unbalanced(self):
        models_conf = os.path.join(ROOT, "test/config_unwrapped_unet_base/test_models.yaml")
        models_conf = OmegaConf.load(models_conf).models
        modellib = MockModelLib()
        model = MockModel(models_conf["TestUnbalanced"], "", None, modellib)
        data = []
        d = model(data, precomputed_up="Hey", precomputed_down="Yay")
        self.assertEqual(d, [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
