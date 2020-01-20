from typing import Dict
import torchnet as tnt
import torch
import numpy as np

from models.base_model import BaseModel
from .base_tracker import BaseTracker


class RegressionTracker(BaseTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False):
        """ This is a generic tracker for regression tasks.
        Use the tracker to track an epoch.
        You can use the reset function before you start a new epoch

        Arguments:
            dataset  -- dataset to track
        Keyword Arguments:
            stage {str} -- current stage. (train, validation, test, etc...) (default: {"train"})
            wandb_log {str} --  Log using weight and biases
        """
        super(RegressionTracker, self).__init__(stage, wandb_log, use_tensorboard)
        self.reset(stage)

    def reset(self, stage="train"):
        super().reset(stage=stage)

    def track(self, model: BaseModel):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)

        outputs = self._convert(model.get_output())
        targets = self._convert(model.get_labels())

        er = torch.sqrt(((outputs - targets) / targets) ** 2)
        self._mer = torch.mean(er).item()

    def get_metrics(self, verbose=False) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)
        metrics["{}_mer".format(self._stage)] = self._mer
        return metrics
