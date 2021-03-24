import os
import sys
from omegaconf import OmegaConf
import torch
import unittest
import logging
from torch_geometric.data import Data

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

from torch_points3d.models.base_model import BaseModel
from test.mock_models import DifferentiableMockModel

log = logging.getLogger(__name__)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class TestLrScheduler(unittest.TestCase):
    def test_update_scheduler_on_epoch(self):

        base_lr = 0.1
        gamma = 0.9
        conf = os.path.join(DIR, "test_config/lr_scheduler.yaml")
        opt = OmegaConf.load(conf)
        opt.update_lr_scheduler_on = "on_epoch"
        model = DifferentiableMockModel(opt)
        model.instantiate_optimizers(opt)
        model.schedulers.__repr__()

        data = Data(pos=torch.randn((1, 3)))
        model.set_input(data, torch.device("cpu"))

        num_epochs = 5
        num_samples_epoch = 32
        batch_size = 4
        steps = num_samples_epoch // batch_size

        for epoch in range(num_epochs):
            for step in range(steps):
                model.optimize_parameters(epoch, batch_size)
        self.assertAlmostEqual(get_lr(model._optimizer), base_lr * gamma ** (num_epochs - 1))

    def test_update_scheduler_on_sample(self):
        base_lr = 0.1
        gamma = 0.9
        conf = os.path.join(DIR, "test_config/lr_scheduler.yaml")
        opt = OmegaConf.load(conf)
        opt.update_lr_scheduler_on = "on_num_sample"
        model = DifferentiableMockModel(opt)
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

        self.assertAlmostEqual(get_lr(model._optimizer), base_lr * gamma ** (num_epochs))

    def test_update_scheduler_on_batch(self):
        base_lr = 0.1
        gamma = 0.9
        conf = os.path.join(DIR, "test_config/lr_scheduler.yaml")
        opt = OmegaConf.load(conf)
        opt.update_lr_scheduler_on = "on_num_batch"
        model = DifferentiableMockModel(opt)
        model.instantiate_optimizers(opt)

        data = Data(pos=torch.randn((1, 3)))
        model.set_input(data, torch.device("cpu"))

        num_epochs = 5
        num_samples_epoch = 32
        batch_size = 4
        steps = num_samples_epoch // batch_size

        count_batch = 0
        for epoch in range(num_epochs):
            for step in range(steps):
                count_batch += 1
                model.optimize_parameters(epoch, batch_size)
        self.assertAlmostEqual(get_lr(model._optimizer), base_lr * gamma ** (count_batch))


if __name__ == "__main__":
    unittest.main()
