from typing import Dict, Any
import torch
import numpy as np

from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.metrics.meters import APMeter
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.models import model_interface


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


class SegTracker(BaseTracker):
    def __init__(
        self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False, ignore_label: int = IGNORE_LABEL
    ):
        """ This is a generic tracker for segmentation tasks.
        It uses a confusion matrix in the back-end to track results.
        Use the tracker to track an epoch.
        You can use the reset function before you start a new epoch

        Arguments:
            dataset  -- dataset to track (used for the number of classes)

        Keyword Arguments:
            stage {str} -- current stage. (train, validation, test, etc...) (default: {"train"})
            wandb_log {str} --  Log using weight and biases
        """
        super(SegTracker, self).__init__(stage, wandb_log, use_tensorboard)
        self._num_classes = dataset.num_classes
        self._ignore_label = ignore_label
        self._dataset = dataset
        self.reset(stage)
        self._metric_func = {
            "miou": max,
            "macc": max,
            "acc": max,
            "loss": min,
            "map": max,
        }  # Those map subsentences to their optimization functions

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._acc = 0
        self._macc = 0
        self._miou = 0
        self._miou_per_class = {}
        self._hist = np.zeros((self._num_classes, self._num_classes))

    @staticmethod
    def detach_tensor(tensor):
        if torch.torch.is_tensor(tensor):
            tensor = tensor.detach()
        return tensor

    def track(self, model: model_interface.TrackerInterface, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        if not self._dataset.has_labels(self._stage):
            return

        super().track(model)

        outputs = model.get_output()
        targets = model.get_labels()
        with torch.no_grad():
            self._compute_metrics(outputs, targets)

    def _compute_metrics(self, outputs, labels):
        pred = outputs.max(1)[1]
        self._hist += fast_hist(pred.cpu().numpy().flatten(), labels.cpu().numpy().flatten(), self._num_classes)
        ious = per_class_iu(self._hist) * 100
        acc = self._hist.diagonal() / self._hist.sum(1) * 100
        self._acc = 0
        self._macc = np.nanmean(acc)
        self._miou = np.nanmean(ious)
        self._miou_per_class = {i: "{:.2f}".format(v) for i, v in enumerate(ious)}

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        metrics["{}_acc".format(self._stage)] = self._acc
        metrics["{}_macc".format(self._stage)] = self._macc
        metrics["{}_miou".format(self._stage)] = self._miou

        if verbose:
            metrics["{}_miou_per_class".format(self._stage)] = self._miou_per_class
        return metrics

    @property
    def metric_func(self):
        return self._metric_func
