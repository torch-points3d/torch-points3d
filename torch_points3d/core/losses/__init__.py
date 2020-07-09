import sys

from .losses import *
from .huber_loss import *
from .panoptic_losses import *
from pytorch_metric_learning.miners import *
from pytorch_metric_learning.losses import *

_custom_losses = sys.modules["torch_points3d.core.losses.losses"]
_torch_metric_learning_losses = sys.modules["pytorch_metric_learning.losses"]
_torch_metric_learning_miners = sys.modules["pytorch_metric_learning.miners"]
_intersection = set(_custom_losses.__dict__) & set(_torch_metric_learning_losses.__dict__)
_intersection = set([module for module in _intersection if not module.startswith("_")])
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
    class_ = getattr(option, "class", None)
    try:
        params = option.params
    except KeyError:
        params = None

    try:
        lparams = option.lparams
    except KeyError:
        lparams = None

    if "loss" in mode:
        cls = getattr(_custom_losses, class_, None)
        if not cls:
            cls = getattr(_torch_metric_learning_losses, class_, None)
            if not cls:
                raise ValueError("loss %s is nowhere to be found" % class_)
    elif mode == "miner":
        cls = getattr(_torch_metric_learning_miners, class_, None)
        if not cls:
            raise ValueError("miner %s is nowhere to be found" % class_)
    else:
        raise NotImplementedError("Cannot instantiate this mode {}".format(mode))

    if params and lparams:
        return cls(*lparams, **params)
    if params:
        return cls(**params)
    if lparams:
        return cls(*params)
    return cls()
