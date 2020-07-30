import os
import torchnet as tnt
import torch
from typing import Dict, Any
import wandb
from torch.utils.tensorboard import SummaryWriter
import logging

from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.models import model_interface

log = logging.getLogger(__name__)


def meter_value(meter, dim=0):
    return float(meter.value()[dim]) if meter.n > 0 else 0.0


class BaseTracker:
    def __init__(self, stage: str, wandb_log: bool, use_tensorboard: bool):
        self._wandb = wandb_log
        self._use_tensorboard = use_tensorboard
        self._tensorboard_dir = os.path.join(os.getcwd(), "tensorboard")
        self._n_iter = 0
        self._finalised = False
        self._conv_type = None

        if self._use_tensorboard:
            log.info(
                "Access tensorboard with the following command <tensorboard --logdir={}>".format(self._tensorboard_dir)
            )
            self._writer = SummaryWriter(log_dir=self._tensorboard_dir)

    def reset(self, stage="train"):
        self._stage = stage
        self._loss_meters = {}
        self._finalised = False

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        metrics = {}
        for key, loss_meter in self._loss_meters.items():
            value = meter_value(loss_meter, dim=0)
            if value:
                metrics[key] = meter_value(loss_meter, dim=0)
        return metrics

    @property
    def metric_func(self):
        self._metric_func = {"loss": min}
        return self._metric_func

    def track(self, model: model_interface.TrackerInterface, **kwargs):
        if self._finalised:
            raise RuntimeError("Cannot track new values with a finalised tracker, you need to reset it first")
        losses = self._convert(model.get_current_losses())
        self._append_losses(losses)

    def finalise(self, *args, **kwargs):
        """ Lifcycle method that is called at the end of an epoch. Use this to compute
        end of epoch metrics.
        """
        self._finalised = True

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

    def publish_to_tensorboard(self, metrics, step):
        for metric_name, metric_value in metrics.items():
            metric_name = "{}/{}".format(metric_name.replace(self._stage + "_", ""), self._stage)
            self._writer.add_scalar(metric_name, metric_value, step)

    @staticmethod
    def _remove_stage_from_metric_keys(stage, metrics):
        new_metrics = {}
        for metric_name, metric_value in metrics.items():
            new_metrics[metric_name.replace(stage + "_", "")] = metric_value
        return new_metrics

    def publish(self, step):
        """ Publishes the current metrics to wandb and tensorboard
        Arguments:
            step: current epoch
        """
        metrics = self.get_metrics()

        if self._wandb:
            wandb.log(metrics, step=step)

        if self._use_tensorboard:
            self.publish_to_tensorboard(metrics, step)

        return {
            "stage": self._stage,
            "epoch": step,
            "current_metrics": self._remove_stage_from_metric_keys(self._stage, metrics),
        }

    def print_summary(self):
        metrics = self.get_metrics(verbose=True)
        log.info("".join(["=" for i in range(50)]))
        for key, value in metrics.items():
            log.info("    {} = {}".format(key, value))
        log.info("".join(["=" for i in range(50)]))

    @staticmethod
    def _dict_to_str(dictionnary):
        string = "{"
        for key, value in dictionnary.items():
            string += "%s: %.2f," % (str(key), value)
        string += "}"
        return string
