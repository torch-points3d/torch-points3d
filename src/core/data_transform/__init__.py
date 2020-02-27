import sys

import torch_geometric.transforms as T
from .transforms import *
from .features import *

_custom_transforms = sys.modules[__name__]
_torch_geometric_transforms = sys.modules["torch_geometric.transforms"]
_intersection = set(_custom_transforms.__dict__) & set(_torch_geometric_transforms.__dict__)
_intersection = [module for module in _intersection if not module.startswith("_")]
if _intersection:
    raise Exception(
        "It seems that you are overiding a transform from pytorch gemetric, \
            this is forbiden, please rename your classes {}".format(
            _intersection
        )
    )


def instantiate_transform(transform_option):
    """ Creates a transform from an OmegaConf dict such as
    transform: GridSampling
        params:
            size: 0.01
    """
    tr_name = transform_option.transform
    try:
        tr_params = transform_option.params
    except KeyError:
        tr_params = None
    try:
        lparams = transform_option.lparams
    except KeyError:
        lparams = None

    cls = getattr(_custom_transforms, tr_name, None)
    if not cls:
        cls = getattr(_torch_geometric_transforms, tr_name, None)
        if not cls:
            raise ValueError("Transform %s is nowhere to be found" % tr_name)

    if tr_params and lparams:
        return cls(*lparams, **tr_params)

    if tr_params:
        return cls(**tr_params)

    if lparams:
        return cls(*lparams)

    return cls()


def instantiate_transforms(transform_options):
    """ Creates a torch_geometric composite transform from an OmegaConf list such as
    - transform: GridSampling
        params:
            size: 0.01
    - transform: NormaliseScale
    """
    transforms = []
    for transform in transform_options:
        transforms.append(instantiate_transform(transform))
    return T.Compose(transforms)
