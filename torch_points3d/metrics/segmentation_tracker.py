from typing import Dict, Any
import torch
import numpy as np

import pytorch_lightning.metrics as ptlm

# from pytorch_lightning.metrics import Accuracy, ConfusionMatrix
from torch_points3d.metrics.base_tracker import LightningBaseTracker


from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.confusion_matrix import compute_average_intersection_union
from torch_points3d.metrics.confusion_matrix import compute_mean_class_accuracy
from torch_points3d.metrics.confusion_matrix import compute_overall_accuracy
from torch_points3d.metrics.confusion_matrix import compute_intersection_union_per_class

from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.metrics.meters import APMeter
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.models import model_interface


class SegmentationTracker(BaseTracker):
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
        super(SegmentationTracker, self).__init__(stage, wandb_log, use_tensorboard)
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
        self._confusion_matrix = ConfusionMatrix(self._num_classes)
        self._acc = 0
        self._macc = 0
        self._miou = 0
        self._miou_per_class = {}

    @staticmethod
    def detach_tensor(tensor):
        if torch.torch.is_tensor(tensor):
            tensor = tensor.detach()
        return tensor

    @property
    def confusion_matrix(self):
        return self._confusion_matrix.confusion_matrix

    def track(self, model: model_interface.TrackerInterface, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        if not self._dataset.has_labels(self._stage):
            return

        super().track(model)

        outputs = model.get_output()
        targets = model.get_labels()
        self._compute_metrics(outputs, targets)

    def _compute_metrics(self, outputs, labels):
        mask = labels != self._ignore_label
        outputs = outputs[mask]
        labels = labels[mask]

        outputs = self._convert(outputs)
        labels = self._convert(labels)

        if len(labels) == 0:
            return

        assert outputs.shape[0] == len(labels)
        self._confusion_matrix.count_predicted_batch(labels, np.argmax(outputs, 1))

        self._acc = 100 * self._confusion_matrix.get_overall_accuracy()
        self._macc = 100 * self._confusion_matrix.get_mean_class_accuracy()
        self._miou = 100 * self._confusion_matrix.get_average_intersection_union()
        self._miou_per_class = {
            i: "{:.2f}".format(100 * v)
            for i, v in enumerate(self._confusion_matrix.get_intersection_union_per_class()[0])
        }

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


class LightningSegmentationTracker(LightningBaseTracker):
    def __init__(self, num_classes: int, ignore_label: int = IGNORE_LABEL, stage: str = "train"):
        super().__init__(stage)
        self._num_classes = num_classes
        self._ignore_label = ignore_label
        self.confusion_matrix_metric = ptlm.ConfusionMatrix(num_classes=num_classes)

    def compute_metrics_from_cm(self, matrix: torch.Tensor):
        acc = compute_overall_accuracy(matrix)
        macc = compute_mean_class_accuracy(matrix)
        miou = compute_average_intersection_union(matrix)
        iou_per_class = compute_intersection_union_per_class(matrix)
        iou_per_class_dict = {i: 100 * v for i, v in enumerate(iou_per_class)}
        return {
            "{}_acc".format(self.stage): 100 * acc,
            "{}_macc".format(self.stage): 100 * macc,
            "{}_miou".format(self.stage): 100 * miou,
            "{}_iou_per_class".format(self.stage): iou_per_class_dict,
        }

    def track(self, model: model_interface.TrackerInterface, **kwargs):
        outputs = model.get_output()
        targets = model.get_labels()
        mask = targets != self._ignore_label
        outputs = outputs[mask]
        targets = targets[mask]
        matrix = self.confusion_matrix_metric(outputs, targets)

        segmentation_metrics = self.compute_metrics_from_cm(matrix)
        return segmentation_metrics

    def _finalise(self):
        matrix = self.confusion_matrix_metric.compute()
        segmentation_metrics = self.compute_metrics_from_cm(matrix)
        return segmentation_metrics

    def reset(self, stage):
        super().reset(stage)
        self.confusion_matrix_metric = ptlm.ConfusionMatrix(num_classes=self._num_classes)
