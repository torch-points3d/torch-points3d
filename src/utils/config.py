import numpy as np
from typing import List
import shutil
import matplotlib.pyplot as plt
import os
from os import path as osp
import torch
from collections import namedtuple
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from .enums import ConvolutionFormat

def launch_wandb(cfg, tags, launch: bool):
    if launch:
        import wandb
        wandb.init(
            project=cfg.wandb.project,
            tags=tags,
            notes=cfg.wandb.notes,
            name=cfg.wandb.name,
            config={"run_path": os.getcwd()},
        )
        shutil.copyfile(
            os.path.join(os.getcwd(), ".hydra/config.yaml"), os.path.join(os.getcwd(), ".hydra/hydra-config.yaml")
        )
        wandb.save(os.path.join(os.getcwd(), ".hydra/hydra-config.yaml"))
        wandb.save(os.path.join(os.getcwd(), ".hydra/overrides.yaml"))

def determine_stage(cfg, has_val_loader):
    """This function is responsible to determine if the best model selection 
       is going to be on the validation or test dataset
       keys: ["test", "val"]
    """
    selection_stage = getattr(cfg, "selection_stage", None)
    if not selection_stage:
        selection_stage = "val" if has_val_loader else "test"
    else:
        if not has_val_loader and selection_stage == "val": 
            raise Exception("Selection stage should be: test")
    return selection_stage

def merge_omega_conf(opt: DictConfig, d: dict):
    """This function allows to merge a OmegaConf DictConfig with a python dictionnary"""
    new_opt = OmegaConf.create(d)
    return OmegaConf.merge(opt, new_opt)

def is_list(entity):
    return isinstance(entity, list) or isinstance(entity, ListConfig)

def is_iterable(entity):
    return isinstance(entity, list) or isinstance(entity, ListConfig) or isinstance(entity, tuple)

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
        format_conf = OmegaConf.create({"conv_type": conv_type.lower()})
        return OmegaConf.merge(cfg_training, format_conf)
