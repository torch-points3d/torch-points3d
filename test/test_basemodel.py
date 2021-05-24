import unittest
import torch
import omegaconf
from omegaconf import OmegaConf, DictConfig
from torch.nn import (
    Sequential,
    Linear as Lin,
    ReLU,
    LeakyReLU,
    BatchNorm1d as BN,
    Dropout,
)
import os
import sys
from torch_geometric.data import Data

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.append(ROOT)

from torch_points3d.models.base_model import BaseModel
from test.mockdatasets import MockDatasetGeometric
from torch_points3d.models.model_factory import instantiate_model


def load_model_config(task, model_type, model_name):
    models_conf = os.path.join(ROOT, "conf/models/{}/{}.yaml".format(task, model_type))

    if omegaconf.__version__ == '1.4.1':
        config =  OmegaConf.load(models_conf)
        config.update("model_name", model_name)
        config.update("data.task", task)
    else:
        config = OmegaConf.create({"models": OmegaConf.load(models_conf)})
        OmegaConf.update(config, "model_name", model_name, merge=True)
        OmegaConf.update(config, "data.task", task)

    return config


def MLP(channels):
    return Sequential(
        *[Sequential(Lin(channels[i - 1], channels[i]), Dropout(0.5), BN(channels[i])) for i in range(1, len(channels))]
    )


class MockModel(BaseModel):
    __REQUIRED_DATA__ = ["x"]
    __REQUIRED_LABELS__ = ["y"]

    def __init__(self):
        super(MockModel, self).__init__(DictConfig({"conv_type": "Dummy"}))

        self._channels = [12, 12, 12, 12]
        self.nn = MLP(self._channels)

    def set_input(self, a):
        self.input = a


class MockModel_(BaseModel):
    __REQUIRED_DATA__ = ["x"]
    __REQUIRED_LABELS__ = ["y"]

    def __init__(self):
        super(MockModel_, self).__init__(DictConfig({"conv_type": "Dummy"}))

        self._channels = [12, 12, 12, 17]
        self.nn = MLP(self._channels)

    def set_input(self, a):
        self.input = a


class TestBaseModel(unittest.TestCase):
    def test_getinput(self):
        model = MockModel()
        with self.assertRaises(AttributeError):
            model.get_input()

        model.set_input(1)
        self.assertEqual(model.get_input(), 1)

    def test_enable_dropout_eval(self):
        model = MockModel()
        model.eval()

        for i in range(len(model._channels) - 1):
            self.assertEqual(model.nn[i][1].training, False)
            self.assertEqual(model.nn[i][2].training, False)

        model.enable_dropout_in_eval()
        for i in range(len(model._channels) - 1):
            self.assertEqual(model.nn[i][1].training, True)
            self.assertEqual(model.nn[i][2].training, False)

    def test_load_pretrained_model(self):
        """
        test load_state_dict_with_same_shape
        """
        model1 = MockModel()
        model2 = MockModel_()

        w1 = model1.state_dict()

        model2.load_state_dict_with_same_shape(w1)
        w2 = model2.state_dict()
        for k, p in w2.items():
            if "nn.2." not in k:
                torch.testing.assert_allclose(w1[k], p)

    def test_accumulated_gradient(self):
        params = load_model_config("segmentation", "pointnet2", "pointnet2ms")
        config_training = OmegaConf.load(os.path.join(DIR, "test_config/training_config.yaml"))
        dataset = MockDatasetGeometric(5)
        model = instantiate_model(params, dataset)
        model.instantiate_optimizers(config_training)
        model.set_input(dataset[0], "cpu")
        expected_make_optimizer_step = [False, False, True, False, False, True, False, False, True, False]
        expected_contains_grads = [False, True, True, False, True, True, False, True, True, False]
        make_optimizer_steps = []
        contains_grads = []
        for epoch in range(10):
            model.forward()

            make_optimizer_step = model._manage_optimizer_zero_grad()  # Accumulate gradient if option is up
            make_optimizer_steps.append(make_optimizer_step)
            grad_ = model._modules["lin1"].weight.grad
            if grad_ is not None:
                contains_grads.append((grad_.sum() != 0).item())
            else:
                contains_grads.append(False)

            model.backward()  # calculate gradients

            if make_optimizer_step:
                model._optimizer.step()  # update parameters

        self.assertEqual(contains_grads, expected_contains_grads)
        self.assertEqual(make_optimizer_steps, expected_make_optimizer_step)

    def test_validatedata(self):
        model = MockModel()
        model.verify_data(Data(x=0), forward_only=True)
        with self.assertRaises(KeyError):
            model.verify_data(Data(x=0))


if __name__ == "__main__":
    unittest.main()
