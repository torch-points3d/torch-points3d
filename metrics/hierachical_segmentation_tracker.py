from typing import Dict
import torchnet as tnt
import numpy as np

from .confusion_matrix import ConfusionMatrix
from .base_tracker import meter_value
from .segmentation_tracker import SegmentationTracker


class HierachicalSegmentationTracker(SegmentationTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False, log_dir: str = ""):
        super(HierachicalSegmentationTracker, self).__init__(dataset, stage, wandb_log, use_tensorboard, log_dir)
        self._class_seg_map = dataset.class_to_segments

    def track(self, losses: Dict[str, float], outputs, targets):
        """ Add current model predictions (usually the result of a batch) to the tracking

        Arguments:
            losses Dict[str,float] -- main loss
            outputs -- model predictions (NxK) where K is the number of labels
            targets -- class labels  - size N
        """
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

        self._acc_meter.add(100 * self._confusion_matrix.get_overall_accuracy())
        self._macc_meter.add(100 * self._confusion_matrix.get_mean_class_accuracy())
        self._miou_meter.add(100 * self._confusion_matrix.get_average_intersection_union())

    def get_metrics(self, verbose=False) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = {}
        for key, loss_meter in self._loss_meters.items():
            metrics[key] = meter_value(loss_meter, dim=0)

        metrics["{}_acc".format(self._stage)] = meter_value(self._acc_meter, dim=0)
        metrics["{}_macc".format(self._stage)] = meter_value(self._macc_meter, dim=0)
        metrics["{}_miou".format(self._stage)] = meter_value(self._miou_meter, dim=0)

        return metrics
