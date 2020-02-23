from typing import Dict
import torchnet as tnt
import torch
import torch.nn.functional as F

from src.models.base_model import BaseModel
from .base_tracker import BaseTracker


class RegressionTracker(BaseTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False, eps=10e-8):
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
        self._eps = eps

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._mer = tnt.meter.AverageValueMeter()
        self._merp = tnt.meter.AverageValueMeter()

    def track(self, model: BaseModel):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)

        outputs = self._convert(model.get_output())
        targets = self._convert(model.get_labels())

        erp = torch.sqrt(((outputs - targets) / (targets + self._eps)) ** 2)
        self._merp.add(torch.mean(erp).item())

        self._mer.add(
            torch.mean(F.normalize(outputs - targets, p=2, dim=-1))
            / torch.mean((F.normalize(targets, p=2, dim=-1) + self._eps))
        ).item()

    def get_metrics(self, verbose=False) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)
        metrics["{}_merp".format(self._stage)] = self._merp.mean
        metrics["{}_mer".format(self._stage)] = self._mer.mean
        return metrics
