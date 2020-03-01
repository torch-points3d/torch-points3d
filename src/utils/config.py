import numpy as np
from typing import List
import shutil
import matplotlib.pyplot as plt
import os
from os import path as osp
import subprocess
import torch
import logging
from collections import namedtuple
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from .enums import ConvolutionFormat
from src.utils.debugging_vars import DEBUGGING_VARS
from src.utils.colors import COLORS, colored_print

log = logging.getLogger(__name__)

def set_debugging_vars_to_global(cfg):
    for key in cfg.keys():
        key_upper = key.upper()
        if key_upper in DEBUGGING_VARS.keys():
            DEBUGGING_VARS[key_upper] = cfg[key]
    log.info(DEBUGGING_VARS)

def set_to_wandb_args(wandb_args, cfg, name):
    var = getattr(cfg.wandb, name, None)
    if var:
        wandb_args[name]=var   

def launch_wandb(cfg, launch: bool):
    if launch:
        import wandb

        model_config = getattr(cfg.models, cfg.model_name, None)
        model_class = getattr(model_config, "class")
        tested_dataset_class = getattr(cfg.data, "class")
        otimizer_class = getattr(cfg.training.optim.optimizer, "class")
        scheduler_class = getattr(cfg.lr_scheduler, "class")
        tags = [
            cfg.model_name,
            model_class.split(".")[0],
            tested_dataset_class,
            otimizer_class,
            scheduler_class,
        ]

        wandb_args = {}
        wandb_args["project"]=cfg.wandb.project
        wandb_args["tags"] = tags
        wandb_args["config"]={"run_path": os.getcwd(),
                              "commit": subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()}
        set_to_wandb_args(wandb_args, cfg, "name")
        set_to_wandb_args(wandb_args, cfg, "entity")
        set_to_wandb_args(wandb_args, cfg, "notes")

        wandb.init(**wandb_args)
        
        shutil.copyfile(
            os.path.join(os.getcwd(), ".hydra/config.yaml"), os.path.join(os.getcwd(), ".hydra/hydra-config.yaml")
        )
        wandb.save(os.path.join(os.getcwd(), ".hydra/hydra-config.yaml"))
        wandb.save(os.path.join(os.getcwd(), ".hydra/overrides.yaml"))

def is_list(entity):
    return isinstance(entity, list) or isinstance(entity, ListConfig)


def is_iterable(entity):
    return isinstance(entity, list) or isinstance(entity, ListConfig) or isinstance(entity, tuple)


def is_dict(entity):
    return isinstance(entity, dict) or isinstance(entity, DictConfig)
