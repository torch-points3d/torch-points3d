from torch.optim import lr_scheduler
from functools import partial
from omegaconf.dictconfig import DictConfig
import logging
from src.utils.config import merge_omega_conf


log = logging.getLogger(__name__)

def repr(self, learning_rate_params={}):
    return "{}({})".format(self.__class__.__name__, learning_rate_params)

class SchedulerWrapper():

    def __init__(self, scheduler, learning_rate_params):
        self._scheduler = scheduler
        self._learning_rate_params = learning_rate_params

    @property
    def scheduler(self):
        return self._scheduler

    def __repr__(self):
        return "{}({})".format(self._scheduler.__class__.__name__, self._learning_rate_params)

    def step(self, *args, **kwargs):
        self._scheduler.step(*args, **kwargs)


def instantiate_scheduler(optimizer, scheduler_opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        scheduler_opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    base_lr = optimizer.defaults['lr']
    learning_rate_params = scheduler_opt.params
    if scheduler_opt.lr_policy == 'lambda_rule':
        if scheduler_opt.rule == "step_decay":
            lr_lambda = lambda e: max(
                learning_rate_params.lr_decay ** (e // learning_rate_params.decay_step),
                learning_rate_params.lr_clip / base_lr,
            )
        elif scheduler_opt.rule == "exponential_decay":
            lr_lambda = lambda e: max(
                eval(learning_rate_params.gamma) ** (e / learning_rate_params.decay_step),
                learning_rate_params.lr_clip / base_lr,
            )            
        else:
            raise NotImplementedError
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    elif scheduler_opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, **learning_rate_params)
    
    elif scheduler_opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **learning_rate_params)
        learning_rate_params = merge_omega_conf(learning_rate_params, {"metric_name": scheduler_opt.metric_name})
        setattr(scheduler, "metric_name", scheduler_opt.metric_name)

    elif scheduler_opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **learning_rate_params)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    
    return SchedulerWrapper(scheduler, learning_rate_params)
