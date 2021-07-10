import sys
from torch.optim import lr_scheduler
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import logging
from torch.optim.lr_scheduler import LambdaLR

from torch_points3d.utils.enums import SchedulerUpdateOn

log = logging.getLogger(__name__)

_custom_lr_scheduler = sys.modules[__name__]


class LambdaStepLR(LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_step=-1):
        super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

    @property
    def last_step(self):
        """Use last_epoch for the step counter"""
        return self.last_epoch

    @last_step.setter
    def last_step(self, v):
        self.last_epoch = v


class PolyLR(LambdaStepLR):
    """DeepLab learning rate policy"""

    def __init__(self, optimizer, max_iter, power=0.9, last_step=-1):
        lambda_func = lambda s: (1 - s / (max_iter + 1)) ** power
        composite_func = lambda s: lambda_func(max_iter) if s > max_iter else lambda_func(s)
        super(PolyLR, self).__init__(optimizer, lambda s: composite_func(s), last_step)


class SquaredLR(LambdaStepLR):
    """ Used for SGD Lars"""

    def __init__(self, optimizer, max_iter, last_step=-1):
        super(SquaredLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1)) ** 2, last_step)


class ExpLR(LambdaStepLR):
    def __init__(self, optimizer, step_size, gamma=0.9, last_step=-1):
        # (0.9 ** 21.854) = 0.1, (0.95 ** 44.8906) = 0.1
        # To get 0.1 every N using gamma 0.9, N * log(0.9)/log(0.1) = 0.04575749 N
        # To get 0.1 every N using gamma g, g ** N = 0.1 -> N * log(g) = log(0.1) -> g = np.exp(log(0.1) / N)
        super(ExpLR, self).__init__(optimizer, lambda s: gamma ** (s / step_size), last_step)


def repr(self, scheduler_params={}):
    return "{}({})".format(self.__class__.__name__, scheduler_params)

