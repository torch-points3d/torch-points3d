from typing import Dict
import torchnet as tnt
import numpy as np

from .confusion_matrix import ConfusionMatrix
from .base_tracker import meter_value
from .segmentation_tracker import SegmentationTracker


class HierarchicalSegmentationTracker(SegmentationTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False, log_dir: str = ""):
        """ Hierarchical metric tracker for Semantic segmentation The dataset needs to have a
         class_to_segment member that defines how metrics get computed and agregated.

        Arguments:
            dataset {[type]}

        Keyword Arguments:
            stage {str} -- current stage (default: {"train"})
            wandb_log {bool} -- Log to Wanndb (default: {False})
            use_tensorboard {bool} -- Log to tensorboard (default: {False})
            log_dir {str} -- Directory for the logs (default: {""})
        """
        super(HierarchicalSegmentationTracker, self).__init__(dataset, stage, wandb_log, use_tensorboard, log_dir)
        self._class_seg_map = dataset.class_to_segments

    def track(self, model):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        losses = model.get_current_losses()
        outputs = model.get_output()
        targets = model.get_labels()
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

        self._acc_per_class, self._macc_per_class, self._miou_per_class = self._get_metrics_per_class()

        self._acc = self._mean_overall(self._acc_per_class.values())
        self._macc = self._mean_overall(self._macc_per_class.values())
        self._miou = self._mean_overall(self._miou_per_class.values())

    def _get_metrics_per_class(self):
        acc = {}
        macc = {}
        miou = {}
        for classname, class_keys in self._class_seg_map.items():
            confusion_for_class = self._confusion_matrix.confusion_matrix[class_keys][:, class_keys]
            confusion_for_class = ConfusionMatrix.create_from_matrix(confusion_for_class)
            acc[classname] = 100 * confusion_for_class.get_overall_accuracy()
            macc[classname] = 100 * confusion_for_class.get_mean_class_accuracy()
            miou[classname] = 100 * confusion_for_class.get_average_intersection_union()
        return acc, macc, miou

    @staticmethod
    def _mean_overall(values_all_classes):
        non_zero = 0
        count = 0
        for value in values_all_classes:
            if value > 0:
                count += value
                non_zero += 1
        if non_zero == 0:
            return 0
        return count / float(non_zero)

    def get_metrics(self, verbose=False) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = {}
        for key, loss_meter in self._loss_meters.items():
            metrics[key] = meter_value(loss_meter, dim=0)

        metrics["{}_acc".format(self._stage)] = self._acc
        metrics["{}_macc".format(self._stage)] = self._macc
        metrics["{}_miou".format(self._stage)] = self._miou

        if verbose:
            metrics["{}_acc_per_class".format(self._stage)] = self._acc_per_class
            metrics["{}_macc_per_class".format(self._stage)] = self._macc_per_class
            metrics["{}_miou_per_class".format(self._stage)] = self._miou_per_class
        return metrics
