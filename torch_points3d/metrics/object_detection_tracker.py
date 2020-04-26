from typing import Dict
import torchnet as tnt
import torch

from torch_points3d.models.base_model import BaseModel
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.metrics.meters import APMeter
from torch_points3d.datasets.segmentation import IGNORE_LABEL


class ObjectDetectionTracker(BaseTracker):
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
        super(ObjectDetectionTracker, self).__init__(stage, wandb_log, use_tensorboard)
        self._num_classes = dataset.num_classes
        self._dataset = dataset
        self.reset(stage)

    def reset(self, stage="train"):
        super().reset(stage=stage)

    @staticmethod
    def detach_tensor(tensor):
        if torch.torch.is_tensor(tensor):
            tensor = tensor.detach()
        return tensor

    @property
    def confusion_matrix(self):
        return self._confusion_matrix.confusion_matrix

    def track(self, model: BaseModel, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)

        outputs = model.get_output()
        targets = model.get_labels()

        obj_pred_val = torch.argmax(outputs["objectness_scores"], 2)  # B,K
        self._obj_acc = torch.sum(
            (obj_pred_val == targets["objectness_label"].long()).float() * targets["objectness_mask"]
        ) / (torch.sum(targets["objectness_mask"]) + 1e-6)

    def get_metrics(self, verbose=False) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        metrics["{}_acc".format(self._stage)] = self._obj_acc.item()

        return metrics

    @property
    def metric_func(self):
        self._metric_func = {
            "acc": max,
        }  # Those map subsentences to their optimization functions
        return self._metric_func
