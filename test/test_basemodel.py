import unittest
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

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.append(ROOT)

from torch_points3d.models.base_model import BaseModel
from test.mockdatasets import MockDatasetGeometric
from torch_points3d.models.model_factory import instantiate_model


def load_model_config(task, model_type, model_name):
    models_conf = os.path.join(ROOT, "conf/models/{}/{}.yaml".format(task, model_type))
    config = OmegaConf.load(models_conf)
    config.update("model_name", model_name)
    config.update("data.task", task)
    return config


def MLP(channels):
    return Sequential(
        *[Sequential(Lin(channels[i - 1], channels[i]), Dropout(0.5), BN(channels[i])) for i in range(1, len(channels))]
    )


class MockModel(BaseModel):
    def __init__(self):
        super(MockModel, self).__init__(DictConfig({"conv_type": "Dummy"}))

        self._channels = [12, 12, 12, 12]
        self.nn = MLP(self._channels)

    def set_input(self, a):
        self.input = a


class TestSimpleBatch(unittest.TestCase):
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


class TestBaseModel(unittest.TestCase):
    def test_getinput(self):
        model = MockModel()
        with self.assertRaises(AttributeError):
            model.get_input()

        model.set_input(1)
        self.assertEqual(model.get_input(), 1)


if __name__ == "__main__":
    unittest.main()
