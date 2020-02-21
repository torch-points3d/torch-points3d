import sys

from .losses import *
from pytorch_metric_learning.miners import *
from pytorch_metric_learning.losses import *

_custom_losses = sys.modules["src.core.losses.losses"]
_torch_metric_learning_losses = sys.modules["pytorch_metric_learning.losses"]
_torch_metric_learning_miners = sys.modules["pytorch_metric_learning.miners"]
_intersection = set(_custom_losses.__dict__) & set(_torch_metric_learning_losses.__dict__)
_intersection = [module for module in _intersection if not module.startswith("_")]
if _intersection:
    raise Exception(
        "It seems that you are overiding a transform from pytorch metric learning, \
            this is forbiden, please rename your classes {}".format(
            _intersection
        )
    )


def instantiate_loss_or_miner(option, mode="loss"):
    """
    create a loss from an OmegaConf dict such as
    TripletMarginLoss.
    params:
        margin=0.1
    It can also instantiate a miner to better learn a loss
    """
    name = option.name
    try:
        params = option.params
    except KeyError:
        params = None

    try:
        lparams = option.lparams
    except KeyError:
        lparams = None

    if mode == "loss":
        cls = getattr(_custom_losses, name, None)
        if not cls:
            cls = getattr(_torch_metric_learning_losses, name, None)
            if not cls:
                raise ValueError("loss %s is nowhere to be found" % name)
    elif mode == "miner":
        cls = getattr(_torch_metric_learning_miners, name, None)
        if not cls:
            raise ValueError("miner %s is nowhere to be found" % name)

    if params and lparams:
        return cls(*lparams, **params)
    if params:
        return cls(**params)
    if lparams:
        return cls(*params)
    return cls()
