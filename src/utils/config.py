
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import os
from os import path as osp
import torch
from collections import namedtuple
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig

from .enums import ConvolutionFormat

def is_list(entity):
    return isinstance(entity, list) or isinstance(entity, ListConfig)


def is_dict(entity):
    return isinstance(entity, dict) or isinstance(entity, DictConfig)

def set_format(model_config, cfg_training):
    """ Adds the type of convolution (DENSE, PARTIAL_DENSE, MESSAGE_PASSING)
    to the training configuration
    """
    conv_type = getattr(model_config, "conv_type", None)
    if conv_type not in [d.name for d in ConvolutionFormat]:
        raise Exception("The format type should be defined within {}".format([d.name for d in ConvolutionFormat]))
    else:
        format_conf = OmegaConf.create(
            {"conv_type": conv_type.lower(), "use_torch_loader": ConvolutionFormat[conv_type].value[1]}
        )
        return OmegaConf.merge(cfg_training, format_conf)


def merges_in_sub(x, list_conf: List):
    dict_ = {}
    for o, v in x.items():
        name = str(o)
        if isinstance(v, DictConfig):
            for c in list_conf:
                v = OmegaConf.merge(v, c)
            dict_[name] = v
        else:
            dict_[name] = v
    return OmegaConf.create(dict_)