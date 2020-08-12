from typing import Dict, Any
import torch
import torchnet as tnt

from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.models import model_interface


class GenerationTracker(BaseTracker):
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
        super(GenerationTracker, self).__init__(stage, wandb_log, use_tensorboard)

    def track(self, model: model_interface.TrackerInterface, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)


