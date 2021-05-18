import sys

import numpy as np
import torch_geometric.transforms as T
from .transforms import *
from .grid_transform import *
from .sparse_transforms import *
from .inference_transforms import *
from .feature_augment import *
from .features import *
from .filters import *
from .precollate import *
from .prebatchcollate import *
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf

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
        # tr_params = transform_option.params
        tr_params = transform_option.get('params')  # Update to OmegaConf 2.0
    except KeyError:
        tr_params = None
    try:
        # lparams = transform_option.lparams
        lparams = transform_option.get('lparams') # Update to OmegaConf 2.0
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


class LotteryTransform(object):
    """
    Transforms which draw a transform randomly among several transforms indicated in transform options
    Examples

    Parameters
    ----------
    transform_options Omegaconf list which contains the transform
    """

    def __init__(self, transform_options):
        self.random_transforms = instantiate_transforms(transform_options)

    def __call__(self, data):

        list_transforms = self.random_transforms.transforms
        i = np.random.randint(len(list_transforms))
        transform = list_transforms[i]
        return transform(data)

    def __repr__(self):
        rep = "LotteryTransform(["
        for trans in self.random_transforms.transforms:
            rep = rep + "{}, ".format(trans.__repr__())
        rep = rep + "])"
        return rep


class ComposeTransform(object):
    """
    Transform to compose other transforms with YAML (Compose of torch_geometric does not work).
    Example :
    .. code-block:: yaml

    - transform: ComposeTransform
      params:
        transform_options:
          - transform: GridSampling3D
            params:
              size: 0.1
          - transform: RandomNoise
            params:
              sigma: 0.05


    Parameters:
    transform_options: Omegaconf Dict
        contains a list of transform
    """
    def __init__(self, transform_options):
        self.transform = instantiate_transforms(transform_options)

    def __call__(self, data):
        return self.transform(data)

    def __repr__(self):
        rep = "ComposeTransform(["
        for trans in self.transform.transforms:
            rep = rep + "{}, ".format(trans.__repr__())
        rep = rep + "])"
        return rep


class RandomParamTransform(object):
    """
    create a transform with random parameters

    Example (on the yaml)

    .. code-block:: yaml

        transform: RandomParamTransform
            params:
                transform_name: GridSampling3D
                transform_params:
                    size:
                        min: 0.1
                        max: 0.3
                        type: "float"
                    mode:
                        value: "last"


    We can also draw random numbers for two parameters, integer or float

    .. code-block:: yaml

        transform: RandomParamTransform
            params:
                transform_name: RandomSphereDropout
                transform_params:
                    radius:
                        min: 1
                        max: 2
                        type: "float"
                    num_sphere:
                        min: 1
                        max: 5
                        type: "int"


    Parameters
    ----------
    transform_name: string:
        the name of the transform
    transform_options: Omegaconf Dict
        contains the name of a variables as a key and min max type as value to specify the range of the parameters and the type of the parameters or it contains the value "value" to specify a variables (see Example above)

    """

    def __init__(self, transform_name, transform_params):
        self.transform_name = transform_name
        self.transform_params = transform_params
        self.random_transform = self._instanciate_transform_with_random_params()

    def _instanciate_transform_with_random_params(self):
        dico = dict()
        for p, rang in self.transform_params.items():
            if "max" in rang and "min" in rang:
                assert rang["max"] - rang["min"] > 0
                v = np.random.random() * (rang["max"] - rang["min"]) + rang["min"]

                if rang["type"] == "float":
                    v = float(v)
                elif rang["type"] == "int":
                    v = int(v)
                else:
                    raise NotImplementedError
                dico[p] = v
            elif "value" in rang:
                v = rang["value"]
                dico[p] = v
            else:
                raise NotImplementedError

        trans_opt = DictConfig(dict(params=dico, transform=self.transform_name))
        random_transform = instantiate_transform(trans_opt, attr="transform")
        return random_transform

    def __call__(self, data):
        self.random_transform = self._instanciate_transform_with_random_params()
        return self.random_transform(data)

    def __repr__(self):
        return "RandomParamTransform({}, params={})".format(self.transform_name, self.transform_params)
