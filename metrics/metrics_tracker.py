import os
from pathlib import Path
from datetime import datetime
from enum import Enum
import numpy as np
import torchnet as tnt
import torch
from typing import Dict
from abc import abstractmethod
import wandb
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from metrics.confusion_matrix import ConfusionMatrix
from metrics.model_checkpoint import ModelCheckpoint
from models.base_model import BaseModel


def get_tracker(model: BaseModel, task: str, dataset, wandb_opt: bool, tensorboard_opt: bool, log_dir: str):
    """Factory method for the tracker

    Arguments:
        task {str} -- task description
        dataset {[type]}
        wandb_log - Log using weight and biases
    Returns:
        [BaseTracker] -- tracker
    """
    tracker = None
    if task.lower() == 'segmentation':
        tracker = SegmentationTracker(dataset, wandb_log=wandb_opt.log,
                                      use_tensorboard=tensorboard_opt.log,
                                      log_dir=log_dir)
    else:
        raise NotImplementedError('No tracker for %s task' % task)

    return tracker


def _meter_value(meter, dim=0):
    return float(meter.value()[dim]) if meter.n > 0 else 0.


class BaseTracker:
    def __init__(self, wandb_log: bool, use_tensorboard: bool, log_dir: str):
        self._wandb = wandb_log
        self._use_tensorboard = use_tensorboard
        self._log_dir = os.path.join(log_dir, "tensorboard")

        self._stage = None
        self._n_iter = 0

        if self._use_tensorboard:
            print("Access tensorboard with the following command <tensorboard --logdir={}>".format(self._log_dir))
            self._writer = SummaryWriter(log_dir=self._log_dir)

    @abstractmethod
    def reset(self, stage="train"):
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def track(self, loss, outputs, targets):
        pass

    def publish_to_tensorboard(self, metrics):
        for metric_name, metric_value in metrics.items():
            metric_name = "{}/{}".format(metric_name.replace(self._stage + "_", ""), self._stage)
            self._writer.add_scalar(metric_name, metric_value, self._n_iter)

    @staticmethod
    def _remove_stage_from_metric_keys(stage, metrics):
        new_metrics = {}
        for metric_name, metric_value in metrics.items():
            new_metrics[metric_name.replace(stage + "_", '')] = metric_value
        return new_metrics

    def publish(self):
        if self._stage == "train":
            self._n_iter += 1

        metrics = self.get_metrics()

        if self._wandb:
            wandb.log(metrics)

        if self._use_tensorboard:
            self.publish_to_tensorboard(metrics)

        return {'stage': self._stage, "epoch": self._n_iter, "current_metrics":
                self._remove_stage_from_metric_keys(self._stage, metrics)}


class SegmentationTracker(BaseTracker):

    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False, log_dir: str = None):
        """ Use the tracker to track an epoch. You can use the reset function before you start a new epoch

        Arguments:
            dataset  -- dataset to track (used for the number of classes)

        Keyword Arguments:
            stage {str} -- current stage. (train, validation, test, etc...) (default: {"train"})
            wandb_log {str} --  Log using weight and biases
        """
        super(SegmentationTracker, self).__init__(wandb_log, use_tensorboard, log_dir)
        self._num_classes = dataset.num_classes
        self._stage = stage

        self.reset(stage)

    def reset(self, stage="train"):
        self._stage = stage

        self._loss_meters = {}
        self._acc_meter = tnt.meter.AverageValueMeter()
        self._macc_meter = tnt.meter.AverageValueMeter()
        self._miou_meter = tnt.meter.AverageValueMeter()
        self._confusion_matrix = ConfusionMatrix(self._num_classes)

    @property
    def confusion_matrix(self):
        return self._confusion_matrix.confusion_matrix

    @staticmethod
    def _convert(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        else:
            return x

    def track(self, losses: Dict[str, float], outputs, targets):
        """ Add current model predictions (usually the result of a batch) to the tracking

        Arguments:
            losses Dict[str,float] -- main loss
            outputs -- model predictions (NxK) where K is the number of labels
            targets -- class labels  - size N
        """
        assert outputs.shape[0] == len(targets)
        for key, loss in losses.items():
            loss_key = '%s_%s' % (self._stage, key)
            if loss_key not in self._loss_meters:
                self._loss_meters[loss_key] = tnt.meter.AverageValueMeter()
            self._loss_meters[loss_key].add(loss)

        outputs = self._convert(outputs)
        targets = self._convert(targets)

        self._confusion_matrix.count_predicted_batch(targets, np.argmax(outputs, 1))

        confusion_matrix_tmp = ConfusionMatrix(self._num_classes)
        confusion_matrix_tmp.count_predicted_batch(targets, np.argmax(outputs, 1))
        self._acc_meter.add(100 * confusion_matrix_tmp.get_overall_accuracy())
        self._macc_meter.add(100 * confusion_matrix_tmp.get_mean_class_accuracy())
        self._miou_meter.add(100 * confusion_matrix_tmp.get_average_intersection_union())

    def get_metrics(self) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = {}
        for key, loss_meter in self._loss_meters.items():
            metrics[key] = _meter_value(loss_meter, dim=0)

        metrics['{}_acc'.format(self._stage)] = _meter_value(self._acc_meter, dim=0)
        metrics['{}_macc'.format(self._stage)] = _meter_value(self._macc_meter, dim=0)
        metrics['{}_miou'.format(self._stage)] = _meter_value(self._miou_meter, dim=0)

        return metrics
