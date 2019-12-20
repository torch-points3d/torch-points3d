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


class COLORS:
    TRAIN_COLOR = '\033[0;92m'
    VAL_COLOR = '\033[0;94m'
    TEST_COLOR = '\033[0;93m'
    BEST_COLOR = '\033[0;92m'


class Coloredtqdm(tqdm):

    def set_postfix(self, ordered_dict=None, refresh=True, color=None, round=4, **kwargs):
        """
        Set/modify postfix (additional stats)
        with automatic formatting based on datatype.

        Parameters
        ----------
        ordered_dict  : dict or OrderedDict, optional
        refresh  : bool, optional
            Forces refresh [default: True].
        kwargs  : dict, optional
        """
        # Sort in alphabetical order to be more deterministic
        postfix = std._OrderedDict([] if ordered_dict is None else ordered_dict)
        for key in sorted(kwargs.keys()):
            postfix[key] = kwargs[key]
        # Preprocess stats according to datatype
        for key in postfix.keys():
            # Number: limit the length of the string
            if isinstance(postfix[key], std.Number):
                postfix[key] = self.format_num_to_k(np.round(postfix[key], round), k=round + 1)
            # Else for any other type, try to get the string conversion
            elif not isinstance(postfix[key], std._basestring):
                postfix[key] = str(postfix[key])
            # Else if it's a string, don't need to preprocess anything
        # Stitch together to get the final postfix
        if color is not None:
            self.postfix = color
        else:
            self.postfix = ''
        self.postfix += ', '.join(key + '=' + postfix[key].strip()
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


def get_tracker(task: str, dataset, wandb_log: bool):
    """Factory method for the tracker

    Arguments:
        task {str} -- task description
        dataset {[type]}
        wandb_log - Log using weight and biases
    Returns:
        [BaseTracker] -- tracker
    """
    if task.lower() == 'segmentation':
        return SegmentationTracker(dataset, wandb_log=wandb_log)
    raise NotImplementedError('No tracker for %s task' % task)


def _meter_value(meter, dim=0):
    return meter.value()[dim] if meter.n > 0 else 0


class BaseTracker:
    def __init__(self,  wandb_log: bool):
        self._wandb = wandb_log

    @abstractmethod
    def reset(self, stage="train"):
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def track(self, loss, outputs, targets):
        pass

    def publish(self):
        if self._wandb:
            wandb.log(self.get_metrics())


class SegmentationTracker(BaseTracker):

    def __init__(self, dataset, stage="train", wandb_log=False):
        """ Use the tracker to track an epoch. You can use the reset function before you start a new epoch

        Arguments:
            dataset  -- dataset to track (used for the number of classes)

        Keyword Arguments:
            stage {str} -- current stage. (train, validation, test, etc...) (default: {"train"})
            wandb_log {str} --  Log using weight and biases
        """
        super(SegmentationTracker, self).__init__(wandb_log)
        self._num_classes = dataset.num_classes
        self._stage = stage
        self._n_iter = 0

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
