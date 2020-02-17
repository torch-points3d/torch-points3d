from torch.optim import lr_scheduler
from omegaconf.dictconfig import DictConfig
import logging
from src.utils.config import merge_omega_conf


log = logging.getLogger(__name__)


def repr(self, scheduler_params={}):
    return "{}({})".format(self.__class__.__name__, scheduler_params)


class LRScheduler:
    def __init__(self, scheduler, scheduler_params):
        self._scheduler = scheduler
        self._scheduler_params = scheduler_params

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def scheduler_opt(self):
        return self._scheduler._scheduler_opt

    def __repr__(self):
        return "{}({})".format(self._scheduler.__class__.__name__, self._scheduler_params)

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

    scheduler_cls_name = getattr(scheduler_opt, "class")
    scheduler_cls = getattr(lr_scheduler, scheduler_cls_name)
    scheduler_params = scheduler_opt.params

    if scheduler_cls_name.lower() == "ReduceLROnPlateau".lower():
        raise NotImplementedError("This scheduler is not fully supported yet")

    scheduler = scheduler_cls(optimizer, **scheduler_params)
    # used to re_create the scheduler
    setattr(scheduler, "_scheduler_opt", scheduler_opt)
    return LRScheduler(scheduler, scheduler_params)
