from torch.optim.lr_scheduler import LambdaLR
from omegaconf.dictconfig import DictConfig


def build_basic_params():
    return DictConfig({"base_lr": 0.001})


def get_scheduler(learning_rate_params, optimizer):
    """ Builds and sets a learning rate scheduler on a given optimizer

    Arguments:
        learning_rate_params -- all parameters required for setting the learning rate
        optimizer -- Optimizer affected by the scheduler

    Returns:
        LRScheduler
    """
    scheduler_builder = _LR_REGISTER.get(learning_rate_params.scheduler_type, None)
    if scheduler_builder is None:
        return None
    return scheduler_builder(learning_rate_params, optimizer)


def _build_step_decay(learning_rate_params, optimizer):
    lr_lbmd = lambda e: max(
        learning_rate_params.lr_decay ** (e // learning_rate_params.decay_step),
        learning_rate_params.lr_clip / learning_rate_params.base_lr,
    )
    return LambdaLR(optimizer, lr_lbmd)


_LR_REGISTER = {"step_decay": _build_step_decay}
