import unittest
from omegaconf import OmegaConf

import os
import sys

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

from torch_points3d.core.schedulers import instantiate_bn_scheduler
from torch_points3d.core.common_modules import MLP
from torch_points3d.core.common_modules.dense_modules import MLP2D


class TestBNMomentumScheduler(unittest.TestCase):
    def test_scheduler(self):
        bn_scheduler_config = OmegaConf.load(os.path.join(DIR, "test_config/bn_scheduler_config.yaml"))
        bn_momentum = bn_scheduler_config.bn_scheduler.params.bn_momentum
        bn_scheduler_params = bn_scheduler_config.bn_scheduler.params
        bn_lambda = lambda e: max(
            bn_scheduler_params.bn_momentum
            * bn_scheduler_params.bn_decay ** (int(e // bn_scheduler_params.decay_step)),
            bn_scheduler_params.bn_clip,
        )
        model = MLP([3, 3, 3], bn_momentum=10)
        bn_scheduler = instantiate_bn_scheduler(model, bn_scheduler_config.bn_scheduler)
        self.assertEqual(model[0][1].batch_norm.momentum, bn_momentum)
        for epoch in range(100):
            bn_scheduler.step(epoch)
            self.assertEqual(model[0][1].batch_norm.momentum, bn_lambda(epoch))

        model = MLP2D([3, 3, 3], bn=True)
        bn_scheduler = instantiate_bn_scheduler(model, bn_scheduler_config.bn_scheduler)
        self.assertEqual(model[0][1].momentum, bn_momentum)
        for epoch in range(100):
            bn_scheduler.step(epoch)
            self.assertEqual(model[0][1].momentum, bn_lambda(epoch))


if __name__ == "__main__":
    unittest.main()
