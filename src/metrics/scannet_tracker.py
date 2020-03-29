from typing import Dict
import torchnet as tnt
import numpy as np
from src.models.base_model import BaseModel
from src.metrics.confusion_matrix import ConfusionMatrix
from src.metrics.base_tracker import BaseTracker, meter_value


class ScannetTracker(BaseTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False, valid_class_ids=None):
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
        super(ScannetTracker, self).__init__(stage, wandb_log, use_tensorboard)
        self._valid_class_ids = np.asarray(valid_class_ids)
        my_dict = {v: i for i, v in enumerate(self._valid_class_ids)}
        self._map_func = np.vectorize(my_dict.get)
        self._num_classes = dataset.num_classes
        self.reset(stage)

    def reset(self, stage="train"):
        super().reset(stage=stage)
        num_classes = len(self._valid_class_ids) + 1 if self._valid_class_ids is not None else self._num_classes
        self._confusion_matrix = ConfusionMatrix(num_classes)

    @property
    def confusion_matrix(self):
        return self._confusion_matrix.confusion_matrix

    @staticmethod
    def map_to_valid(tensor, mapping_func, list_valid_ids):
        valid_ids = np.in1d(tensor, list_valid_ids)
        tensor[valid_ids] = mapping_func(tensor[valid_ids])
        tensor[~valid_ids] = len(list_valid_ids)
        return tensor

    def track(self, model: BaseModel):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)

        outputs = self._convert(model.get_output())
        targets = self._convert(model.get_labels())
        assert outputs.shape[0] == len(targets)

        targets = ScannetTracker.map_to_valid(targets, self._map_func, self._valid_class_ids)
        outputs = ScannetTracker.map_to_valid(np.argmax(outputs, 1), self._map_func, self._valid_class_ids)

        self._confusion_matrix.count_predicted_batch(targets, outputs)

        self._acc = 100 * self._confusion_matrix.get_overall_accuracy()
        self._macc = 100 * self._confusion_matrix.get_mean_class_accuracy()
        self._miou = 100 * self._confusion_matrix.get_average_intersection_union()

    def get_metrics(self, verbose=False) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        metrics["{}_acc".format(self._stage)] = self._acc
        metrics["{}_macc".format(self._stage)] = self._macc
        metrics["{}_miou".format(self._stage)] = self._miou

        return metrics
