from typing import Dict, Any
import torch
import torchnet as tnt

from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.models import model_interface


class ClassificationTracker(BaseTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False):
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
        super(ClassificationTracker, self).__init__(stage, wandb_log, use_tensorboard)
        self.reset(stage)

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._acc = tnt.meter.AverageValueMeter()

    @staticmethod
    def detach_tensor(tensor):
        if torch.torch.is_tensor(tensor):
            tensor = tensor.detach()
        return tensor

    @staticmethod
    def compute_acc(y_hat, y):
        labels_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        return acc

    def track(self, model: model_interface.TrackerInterface, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)

        outputs = model.get_output()
        targets = model.get_labels().flatten()

        self._acc.add(100 * self.compute_acc(outputs, targets))

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)
        metrics["{}_acc".format(self._stage)] = meter_value(self._acc)
        return metrics

    @property
    def metric_func(self):
        self._metric_func = {
            "acc": max,
        }  # Those map subsentences to their optimization functions
        return self._metric_func
