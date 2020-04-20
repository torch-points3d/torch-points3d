import unittest
from omegaconf import DictConfig
from torch.nn import (
    Sequential as Seq,
    Linear as Lin,
    ReLU,
    LeakyReLU,
    BatchNorm1d as BN,
    Dropout,
)
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)

from torch_points3d.models.base_model import BaseModel


def MLP(channels):
    return Seq(
        *[Seq(Lin(channels[i - 1], channels[i]), Dropout(0.5), BN(channels[i])) for i in range(1, len(channels))]
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


class TestBaseModel(unittest.TestCase):
    def test_getinput(self):
        model = MockModel()
        with self.assertRaises(AttributeError):
            model.get_input()

        model.set_input(1)
        self.assertEqual(model.get_input(), 1)


if __name__ == "__main__":
    unittest.main()
