from typing import *
from omegaconf import OmegaConf
from torch import nn
import logging

try:
    import MinkowskiEngine as ME

    BATCH_NORM_MODULES: Any = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        ME.MinkowskiBatchNorm,
        ME.MinkowskiInstanceNorm,
    )
except:
    BATCH_NORM_MODULES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


log = logging.getLogger(__name__)


def set_bn_momentum_default(bn_momentum):
    """
    This function return a function which will assign `bn_momentum` to every module instance within `BATCH_NORM_MODULES`.
    """

    def fn(m):
        if isinstance(m, BATCH_NORM_MODULES):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):
    def __init__(self, model, update_scheduler_on, bn_momentum, bn_decay, decay_step, bn_clip, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(type(model).__name__))

        bn_lambda = lambda e: max(
            bn_momentum
            * bn_decay ** (int(e // decay_step)),
            bn_clip,
        )

        self.model = model
        self.setter = setter
        self.bn_lambda = bn_lambda
        self._current_momemtum = None
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch
        self._update_scheduler_on = update_scheduler_on

    @property
    def update_scheduler_on(self):
        return self._update_scheduler_on

    def step(self, epoch=None):

        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        new_momemtum = self.bn_lambda(epoch)
        if self._current_momemtum != new_momemtum:
            self._current_momemtum = new_momemtum
            log.info("Setting batchnorm momentum at {}".format(new_momemtum))
            self.model.apply(self.setter(new_momemtum))

    def state_dict(self):
        return {
            "current_momemtum": self.bn_lambda(self.last_epoch),
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict["last_epoch"]
        self.current_momemtum = state_dict["current_momemtum"]

    def __repr__(self):
        return "{}(base_momentum: {}, update_scheduler_on={})".format(
            self.__class__.__name__, self._current_momemtum, self._update_scheduler_on
        )
