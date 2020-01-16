import os
from pathlib import Path
from datetime import datetime
import importlib
from enum import Enum
import numpy as np
import torchnet as tnt
import torch
from typing import Dict
from abc import abstractmethod
import wandb
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import logging

from metrics.confusion_matrix import ConfusionMatrix
from metrics.model_checkpoint import ModelCheckpoint
from models.base_model import BaseModel

log = logging.getLogger(__name__)


def get_tracker(
    model: BaseModel, task: str, dataset, wandb_opt: bool, tensorboard_opt: bool, log_dir: str,
):
    """Factory method for the tracker

    Arguments:
        task {str} -- task description
        dataset {[type]}
        wandb_log - Log using weight and biases
    Returns:
        [BaseTracker] -- tracker
    """
    tracker_name = task + "_tracker"
    if dataset.is_hierarchical:
        tracker_name = "hierarchical_" + tracker_name
    tracker_filename = ".".join(["metrics", tracker_name])
    trackerlib = importlib.import_module(tracker_filename)
    cls_name = "".join(tracker_name.split("_"))

    tracker = None
    for name, cls in trackerlib.__dict__.items():
        if name.lower() == cls_name.lower():
            tracker = cls

    if tracker is None:
        raise NotImplementedError("No tracker for %s task" % task)

    return tracker(dataset, wandb_log=wandb_opt.log, use_tensorboard=tensorboard_opt.log, log_dir=log_dir)


def meter_value(meter, dim=0):
    return float(meter.value()[dim]) if meter.n > 0 else 0.0


class BaseTracker:
    def __init__(self, wandb_log: bool, use_tensorboard: bool, log_dir: str):
        self._wandb = wandb_log
        self._use_tensorboard = use_tensorboard
        self._log_dir = os.path.join(log_dir, "tensorboard")

        self._stage = None
        self._n_iter = 0

        if self._use_tensorboard:
            log.info("Access tensorboard with the following command <tensorboard --logdir={}>".format(self._log_dir))
            self._writer = SummaryWriter(log_dir=self._log_dir)

    @abstractmethod
    def reset(self, stage="train"):
        pass

    @abstractmethod
    def get_metrics(self, verbose=False) -> Dict[str, float]:
        pass

    @abstractmethod
    def track(self, model):
        pass

    def publish_to_tensorboard(self, metrics):
        for metric_name, metric_value in metrics.items():
            metric_name = "{}/{}".format(metric_name.replace(self._stage + "_", ""), self._stage)
            self._writer.add_scalar(metric_name, metric_value, self._n_iter)

    @staticmethod
    def _remove_stage_from_metric_keys(stage, metrics):
        new_metrics = {}
        for metric_name, metric_value in metrics.items():
            new_metrics[metric_name.replace(stage + "_", "")] = metric_value
        return new_metrics

    def publish(self):
        if self._stage == "train":
            self._n_iter += 1

        metrics = self.get_metrics()

        if self._wandb:
            wandb.log(metrics)

        if self._use_tensorboard:
            self.publish_to_tensorboard(metrics)

        return {
            "stage": self._stage,
            "epoch": self._n_iter,
            "current_metrics": self._remove_stage_from_metric_keys(self._stage, metrics),
        }

    def print_summary(self):
        metrics = self.get_metrics(verbose=True)
        print("".join(["=" for i in range(50)]))
        for key, value in metrics.items():
            print("    {} = {}".format(key, value))
        print("".join(["=" for i in range(50)]))
