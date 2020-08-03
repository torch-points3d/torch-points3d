import sys

import torch_geometric.transforms as T
from .transforms import *
from .grid_transform import *
from .sparse_transforms import *
from .inference_transforms import *
from .feature_augment import *
from .features import *
from .filters import *
from .precollate import *

_custom_transforms = sys.modules[__name__]
_torch_geometric_transforms = sys.modules["torch_geometric.transforms"]
_intersection_names = set(_custom_transforms.__dict__) & set(_torch_geometric_transforms.__dict__)
_intersection_names = set([module for module in _intersection_names if not module.startswith("_")])
L_intersection_names = len(_intersection_names) > 0
_intersection_cls = []

for transform_name in _intersection_names:
    transform_cls = getattr(_custom_transforms, transform_name)
    if not "torch_geometric.transforms." in str(transform_cls):
        _intersection_cls.append(transform_cls)
L_intersection_cls = len(_intersection_cls) > 0

if L_intersection_names:
    if L_intersection_cls:
        raise Exception(
            "It seems that you are overiding a transform from pytorch gemetric, \
                this is forbiden, please rename your classes {} from {}".format(
                _intersection_names, _intersection_cls
            )
        )
    else:
        raise Exception(
            "It seems you are importing transforms {} from pytorch geometric within the current code base. \
             Please, remove them or add them within a class, function, etc.".format(
                _intersection_names
            )
        )


def instantiate_transform(transform_option, attr="transform"):
    """ Creates a transform from an OmegaConf dict such as
    transform: GridSampling3D
        params:
            size: 0.01
    """
    tr_name = getattr(transform_option, attr, None)
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
    - transform: GridSampling3D
        params:
            size: 0.01
    - transform: NormaliseScale
    """
    transforms = []
    for transform in transform_options:
        transforms.append(instantiate_transform(transform))
    return T.Compose(transforms)


def instantiate_filters(filter_options):
    filters = []
    for filt in filter_options:
        filters.append(instantiate_transform(filt, "filter"))
    return FCompose(filters)
