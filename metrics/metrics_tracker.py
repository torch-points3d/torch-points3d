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

from metrics.confusion_matrix import ConfusionMatrix
from tqdm import tqdm, std

HAS_TENSORBOARD_INSTALLED = False
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD_INSTALLED = True
except:
    pass


class COLORS:
    TRAIN_COLOR = '\033[0;92m'
    VAL_COLOR = '\033[0;94m'
    TEST_COLOR = '\033[0;93m'
    BEST_COLOR = '\033[0;92m'


class Coloredtqdm(tqdm):

    def set_postfix(self, ordered_dict=None, refresh=True, color=None, round=4, **kwargs):
        postfix = std._OrderedDict([] if ordered_dict is None else ordered_dict)

        for key in sorted(kwargs.keys()):
            postfix[key] = kwargs[key]

        for key in postfix.keys():
            if isinstance(postfix[key], std.Number):
                postfix[key] = self.format_num_to_k(np.round(postfix[key], round), k=round + 1)
            if isinstance(postfix[key], std._basestring):
                postfix[key] = str(postfix[key])
            if len(postfix[key]) != round:
                postfix[key] += (round - len(postfix[key])) * " "

        if color is not None:
            self.postfix = color
        else:
            self.postfix = ''

        self.postfix += ', '.join(key + '=' + postfix[key]
                                  for key in postfix.keys())
        if color is not None:
            self.postfix += '\033[0m'

        if refresh:
            self.refresh()

    def format_num_to_k(self, seq, k=4):
        seq = str(seq)
        length = len(seq)
        out = seq + ' ' * (k - length) if length < k else seq
        return out if length < k else seq[:k]


def get_tracker(task: str, dataset, wandb_log: bool, use_tensorboard: bool, log_dir: str):
    """Factory method for the tracker

    Arguments:
        task {str} -- task description
        dataset {[type]}
        wandb_log - Log using weight and biases
    Returns:
        [BaseTracker] -- tracker
    """
    if task.lower() == 'segmentation':
        return SegmentationTracker(dataset, wandb_log=wandb_log, use_tensorboard=use_tensorboard, log_dir=log_dir)
    raise NotImplementedError('No tracker for %s task' % task)


def _meter_value(meter, dim=0):
    return float(meter.value()[dim]) if meter.n > 0 else 0.


class BaseTracker:
    def __init__(self,  wandb_log: bool, use_tensorboard: bool, log_dir: str = 'logs'):
        self._wandb = wandb_log
        self._use_tensorboard = use_tensorboard
        self._stage = None
        self._n_iter = 0

        if self._use_tensorboard and HAS_TENSORBOARD_INSTALLED:
            dirname = Path(os.path.abspath(__file__)).parent.parent
            self._log_dir = os.path.join(dirname, log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
            print("Find tensorboard metrics with the command <tensorboard --logdir={}>".format(self._log_dir))
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
        if self._stage == "train":
            self._n_iter += 1

        for metric_name, metric_value in metrics.items():
            metric_name = "{}/{}".format(metric_name.replace(self._stage+"_", ""), self._stage)
                    
            if self._use_tensorboard and HAS_TENSORBOARD_INSTALLED:
                self._writer.add_scalar(metric_name, metric_value, self._n_iter)

    def publish(self):
        if self._wandb:
            wandb.log(self.get_metrics())


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

        if self._use_tensorboard:
            self.publish_to_tensorboard(metrics)
        return metrics
