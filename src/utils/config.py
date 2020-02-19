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


def is_list(entity):
    return isinstance(entity, list) or isinstance(entity, ListConfig)


def is_iterable(entity):
    return isinstance(entity, list) or isinstance(entity, ListConfig) or isinstance(entity, tuple)


def is_dict(entity):
    return isinstance(entity, dict) or isinstance(entity, DictConfig)
