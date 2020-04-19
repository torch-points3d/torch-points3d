import sys
from torch.optim import lr_scheduler
from omegaconf.dictconfig import DictConfig
import logging
from torch.optim.lr_scheduler import LambdaLR

from torch_points3d.utils.enums import SchedulerUpdateOn

log = logging.getLogger(__name__)

_custom_lr_scheduler = sys.modules[__name__]


def collect_params(params, update_scheduler_on):
    """
    This function enable to handle if params contains on_epoch and on_iter or not.
    """
    on_epoch_params = params.get("on_epoch")
    on_batch_params = params.get("on_num_batch")
    on_sample_params = params.get("on_num_sample")

    def check_params(params):
        if params is not None:
            return params
        else:
            raise Exception(
                "The lr_scheduler doesn't have policy {}. Options: {}".format(update_scheduler_on, SchedulerUpdateOn)
            )

    if on_epoch_params or on_batch_params or on_sample_params:
        if update_scheduler_on == SchedulerUpdateOn.ON_EPOCH.value:
            return check_params(on_epoch_params)
        elif update_scheduler_on == SchedulerUpdateOn.ON_NUM_BATCH.value:
            return check_params(on_batch_params)
        elif update_scheduler_on == SchedulerUpdateOn.ON_NUM_SAMPLE.value:
            return check_params(on_sample_params)
        else:
            raise Exception(
                "The provided update_scheduler_on {} isn't within {}".format(update_scheduler_on, SchedulerUpdateOn)
            )
    else:
        return params


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


class LRScheduler:
    def __init__(self, scheduler, scheduler_params, update_scheduler_on):
        self._scheduler = scheduler
        self._scheduler_params = scheduler_params
        self._update_scheduler_on = update_scheduler_on

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def scheduler_opt(self):
        return self._scheduler._scheduler_opt

    def __repr__(self):
        return "{}({}, update_scheduler_on={})".format(
            self._scheduler.__class__.__name__, self._scheduler_params, self._update_scheduler_on
        )

    def step(self, *args, **kwargs):
        self._scheduler.step(*args, **kwargs)

    def state_dict(self):
        return self._scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self._scheduler.load_state_dict(state_dict)


def instantiate_scheduler(optimizer, scheduler_opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        scheduler_opt (option class) -- dict containing all the params to build the scheduler　
                              opt.lr_policy is the name of learning rate policy: lambda_rule | step | plateau | cosine
                              opt.params contains the scheduler_params to construct the scheduler
    See https://pytorch.org/docs/stable/optim.html for more details.
    """

    update_scheduler_on = scheduler_opt.update_scheduler_on
    scheduler_cls_name = getattr(scheduler_opt, "class")
    scheduler_params = collect_params(scheduler_opt.params, update_scheduler_on)

    try:
        scheduler_cls = getattr(lr_scheduler, scheduler_cls_name)
    except:
        scheduler_cls = getattr(_custom_lr_scheduler, scheduler_cls_name)
        log.info("Created custom lr scheduler")

    if scheduler_cls_name.lower() == "ReduceLROnPlateau".lower():
        raise NotImplementedError("This scheduler is not fully supported yet")

    scheduler = scheduler_cls(optimizer, **scheduler_params)
    # used to re_create the scheduler

    setattr(scheduler, "_scheduler_opt", scheduler_opt)
    return LRScheduler(scheduler, scheduler_params, update_scheduler_on)
