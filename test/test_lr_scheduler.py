import os
import sys
from omegaconf import OmegaConf
import torch
from torch import nn
import unittest
import logging
from torch_geometric.data import Data

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

from src.models.base_model import BaseModel

log = logging.getLogger(__name__)


class MockModel(BaseModel):
    def __init__(self, opt):
        super(MockModel, self).__init__(opt)

        self.nn = nn.Linear(3, 3)

    def set_input(self, data, device):
        self.pos = data.pos

    def forward(self):
        self.output = self.nn(self.pos)
        self.loss = self.output.sum()

    def backward(self):
        self.loss.backward()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class TestLrScheduler(unittest.TestCase):
    def test_update_scheduler_on_epoch(self):

        base_lr = 0.1
        gamma = 0.9
        conf = os.path.join(DIR, "test_config/lr_scheduler.yaml")
        opt = OmegaConf.load(conf)
        opt.update_scheduler_on_epoch = True
        model = MockModel(opt)
        model.instantiate_optimizers(opt)

        data = Data(pos=torch.randn((1, 3)))
        model.set_input(data, torch.device("cpu"))

        num_epochs = 5
        num_samples_epoch = 32
        batch_size = 4
        steps = num_samples_epoch // batch_size

        for epoch in range(num_epochs):
            for step in range(steps):
                model.optimize_parameters(epoch, batch_size)
        self.assertEqual(get_lr(model._optimizer), base_lr * gamma ** (num_epochs - 1))

    def test_update_scheduler_on_iter(self):
        base_lr = 0.1
        gamma = 0.9
        conf = os.path.join(DIR, "test_config/lr_scheduler.yaml")
        opt = OmegaConf.load(conf)
        opt.update_scheduler_on_epoch = False
        model = MockModel(opt)
        model.instantiate_optimizers(opt)

        data = Data(pos=torch.randn((1, 3)))
        model.set_input(data, torch.device("cpu"))

        num_epochs = 5
        num_samples_epoch = 32
        batch_size = 4
        steps = num_samples_epoch // batch_size

        for epoch in range(num_epochs):
            for step in range(steps):
                model.optimize_parameters(epoch, batch_size)

        self.assertEqual(get_lr(model._optimizer), base_lr * gamma ** (num_epochs - 1))


if __name__ == "__main__":
    unittest.main()
