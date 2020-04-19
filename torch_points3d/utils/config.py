import numpy as np
from typing import List
import shutil
import matplotlib.pyplot as plt
import os
from os import path as osp
import torch
import logging
from collections import namedtuple
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from .enums import ConvolutionFormat
from torch_points3d.utils.debugging_vars import DEBUGGING_VARS
from torch_points3d.utils.colors import COLORS, colored_print
import subprocess

log = logging.getLogger(__name__)


class ConvolutionFormatFactory:
    @staticmethod
    def check_is_dense_format(conv_type):
        if (
            conv_type.lower() == ConvolutionFormat.PARTIAL_DENSE.value.lower()
            or conv_type.lower() == ConvolutionFormat.MESSAGE_PASSING.value.lower()
            or conv_type.lower() == ConvolutionFormat.SPARSE.value.lower()
        ):
            return False
        elif conv_type.lower() == ConvolutionFormat.DENSE.value.lower():
            return True
        else:
            raise NotImplementedError("Conv type {} not supported".format(conv_type))


class Option:
    """This class is used to enable accessing arguments as attributes without having OmaConf.
       It is used along convert_to_base_obj function
    """

    def __init__(self, opt):
        for key, value in opt.items():
            setattr(self, key, value)


def convert_to_base_obj(opt):
    return Option(OmegaConf.to_container(opt))


def set_debugging_vars_to_global(cfg):
    for key in cfg.keys():
        key_upper = key.upper()
        if key_upper in DEBUGGING_VARS.keys():
            DEBUGGING_VARS[key_upper] = cfg[key]
    log.info(DEBUGGING_VARS)


def set_to_wandb_args(wandb_args, cfg, name):
    var = getattr(cfg.wandb, name, None)
    if var:
        wandb_args[name] = var


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
        wandb_args["project"] = cfg.wandb.project
        wandb_args["tags"] = tags
        try:
            wandb_args["config"] = {
                "run_path": os.getcwd(),
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip(),
            }
        except:
            wandb_args["config"] = {
                "run_path": os.getcwd(),
                "commit": "n/a",
            }
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
