from typing import Dict
import torchnet as tnt
import torch

from torch_points3d.models.model_interface import TrackerInterface
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.metrics.meters import APMeter
from torch_points3d.datasets.segmentation import IGNORE_LABEL

from torch_points3d.modules.VoteNet import VoteNetResults


class ObjectDetectionTracker(BaseTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False):
        super(ObjectDetectionTracker, self).__init__(stage, wandb_log, use_tensorboard)
        self._num_classes = dataset.num_classes
        self._dataset = dataset
        self.reset(stage)
        self._metric_func = {"loss": min, "acc": max}

    def reset(self, stage="train"):
        super().reset(stage=stage)

    @staticmethod
    def detach_tensor(tensor):
        if torch.torch.is_tensor(tensor):
            tensor = tensor.detach()
        return tensor

    def track(self, model: TrackerInterface, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)

        outputs: VoteNetResults = model.get_output()

        total_num_proposal = outputs.objectness_label.shape[0] * outputs.objectness_label.shape[1]
        self._pos_ratio = torch.sum(outputs.objectness_label.float()) / float(total_num_proposal)
        self._neg_ratio = torch.sum(outputs.objectness_label.float()) / float(total_num_proposal) - self._pos_ratio

        obj_pred_val = torch.argmax(outputs.objectness_scores, 2)  # B,K
        self._obj_acc = torch.sum(
            (obj_pred_val == outputs.objectness_label.long()).float() * outputs.objectness_mask
        ) / (torch.sum(outputs.objectness_mask) + 1e-6)

    def get_metrics(self, verbose=False) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        metrics["{}_acc".format(self._stage)] = self._obj_acc.item()

        return metrics

    @property
    def metric_func(self):
        return self._metric_func
