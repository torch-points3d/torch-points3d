from typing import Dict
import torchnet as tnt
import torch
import numpy as np

from models.base_model import BaseModel
from .confusion_matrix import ConfusionMatrix
from .base_tracker import BaseTracker, meter_value


class SegmentationTracker(BaseTracker):
    def __init__(
        self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False, log_dir: str = "",
    ):
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

    def _ingest_data(self, model: BaseModel):
        """ Internal method for extracting the metrics from the model and adding those
        to the loss tracker and confusion matrix
        """
        losses = model.get_current_losses()
        outputs = model.get_output()
        targets = model.get_labels()
        batch_idx = model.get_batch_idx()
        assert outputs.shape[0] == len(targets)

        for key, loss in losses.items():
            if loss is None:
                continue
            loss_key = "%s_%s" % (self._stage, key)
            if loss_key not in self._loss_meters:
                self._loss_meters[loss_key] = tnt.meter.AverageValueMeter()
            self._loss_meters[loss_key].add(loss)

        outputs = self._convert(outputs)
        targets = self._convert(targets)

        self._confusion_matrix.count_predicted_batch(targets, np.argmax(outputs, 1))

    def track(self, model: BaseModel):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        self._acc = 100 * self._confusion_matrix.get_overall_accuracy()
        self._macc = 100 * self._confusion_matrix.get_mean_class_accuracy()
        self._miou = 100 * self._confusion_matrix.get_average_intersection_union()

    def get_metrics(self, verbose=False) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = {}
        for key, loss_meter in self._loss_meters.items():
            metrics[key] = meter_value(loss_meter, dim=0)

        metrics["{}_acc".format(self._stage)] = self._acc
        metrics["{}_macc".format(self._stage)] = self._macc
        metrics["{}_miou".format(self._stage)] = self._miou

        return metrics
