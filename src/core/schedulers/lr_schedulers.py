from torch.optim import lr_scheduler
from functools import partial
from omegaconf.dictconfig import DictConfig
import logging
from src.utils.config import merge_omega_conf


log = logging.getLogger(__name__)

def repr(self, scheduler_params={}):
    return "{}({})".format(self.__class__.__name__, scheduler_params)

class SchedulerWrapper():

    def __init__(self, scheduler, scheduler_params):
        self._scheduler = scheduler
        self._scheduler_params = scheduler_params

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def scheduler_opt(self):
        return  self._scheduler._scheduler_opt

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
    base_lr = optimizer.defaults['lr']
    scheduler_params = scheduler_opt.params
    if scheduler_opt.lr_policy == 'lambda_rule':
        if scheduler_opt.rule == "step_decay":
            lr_lambda = lambda e: max(
                scheduler_params.lr_decay ** (e // scheduler_params.decay_step),
                scheduler_params.lr_clip / base_lr,
            )
        elif scheduler_opt.rule == "exponential_decay":
            lr_lambda = lambda e: max(
                eval(scheduler_params.gamma) ** (e / scheduler_params.decay_step),
                scheduler_params.lr_clip / base_lr,
            )            
        else:
            raise NotImplementedError
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    elif scheduler_opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, **scheduler_params)
    
    elif scheduler_opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
        scheduler_params = merge_omega_conf(scheduler_params, {"metric_name": scheduler_opt.metric_name})
        setattr(scheduler, "metric_name", scheduler_opt.metric_name)

    elif scheduler_opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', scheduler_opt.lr_policy)
    
    # used to re_create the scheduler
    setattr(scheduler, "_scheduler_opt", scheduler_opt)

    return SchedulerWrapper(scheduler, scheduler_params)
