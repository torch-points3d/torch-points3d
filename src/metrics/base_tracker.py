import os
import torchnet as tnt
import torch
from typing import Dict
import wandb
from torch.utils.tensorboard import SummaryWriter
import logging

from src.metrics.confusion_matrix import ConfusionMatrix
from src.metrics.model_checkpoint import ModelCheckpoint
from src.models.base_model import BaseModel

log = logging.getLogger(__name__)


def meter_value(meter, dim=0):
    return float(meter.value()[dim]) if meter.n > 0 else 0.0


class BaseTracker:
    def __init__(self, stage: str, wandb_log: bool, use_tensorboard: bool):
        self._wandb = wandb_log
        self._use_tensorboard = use_tensorboard
        self._tensorboard_dir = os.path.join(os.getcwd(), "tensorboard")
        self._n_iter = 0

        if self._use_tensorboard:
            log.info(
                "Access tensorboard with the following command <tensorboard --logdir={}>".format(self._tensorboard_dir)
            )
            self._writer = SummaryWriter(log_dir=self._tensorboard_dir)

    def reset(self, stage="train"):
        self._stage = stage
        self._loss_meters = {}

    def get_metrics(self, verbose=False) -> Dict[str, float]:
        metrics = {}
        for key, loss_meter in self._loss_meters.items():
            value = meter_value(loss_meter, dim=0)
            if value:
                metrics[key] = meter_value(loss_meter, dim=0)
        return metrics

    def track(self, model):
        losses = self._convert(model.get_current_losses())
        self._append_losses(losses)

    def _append_losses(self, losses):
        for key, loss in losses.items():
            if loss is None:
                continue
            loss_key = "%s_%s" % (self._stage, key)
            if loss_key not in self._loss_meters:
                self._loss_meters[loss_key] = tnt.meter.AverageValueMeter()
            self._loss_meters[loss_key].add(loss)

    @staticmethod
    def _convert(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        else:
            return x

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
        log.info("".join(["=" for i in range(50)]))
        for key, value in metrics.items():
            log.info("    {} = {}".format(key, value))
        log.info("".join(["=" for i in range(50)]))
