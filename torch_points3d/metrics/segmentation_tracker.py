from typing import Dict
import torch
import numpy as np
from torch_geometric.nn.unpool import knn_interpolate

from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.metrics.meters import APMeter
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.models import model_interface


class SegmentationFullResHelpers:
    def __init__(self, raw_data, num_classes, conv_type, class_seg_map=None):
        self._raw_data = raw_data
        self._num_pos = raw_data.pos.shape[0]
        self._votes = torch.zeros((self._num_pos, num_classes), dtype=torch.float)
        self._vote_counts = torch.zeros(self._num_pos, dtype=torch.float)
        self._full_res_preds = None
        self._conv_type = conv_type
        self._class_seg_map = class_seg_map

    @property
    def full_res_labels(self):
        return self._raw_data.y

    @property
    def full_res_preds(self):
        self._predict_full_res()
        if self._class_seg_map:
            return self._full_res_preds[:, self._class_seg_map].argmax(1) + self._class_seg_map[0]
        else:
            return self._full_res_preds.argmax(-1)

    def add_vote(self, data, output, batch_mask):
        """ Populates scores for the points in data

        Parameters
        ----------
        data : Data
            should contain `pos` and `SaveOriginalPosId.KEY` keys
        output : torch.Tensor
            probablities out of the model, shape: [N,nb_classes]
        """
        idx = data[SaveOriginalPosId.KEY][batch_mask]
        self._votes[idx] += output
        self._vote_counts[idx] += 1

    def _predict_full_res(self):
        """ Predict full resolution results based on votes """
        has_prediction = self._vote_counts > 0
        self._votes[has_prediction] /= self._vote_counts[has_prediction].unsqueeze(-1)

        # Upsample and predict
        full_pred = knn_interpolate(
            self._votes[has_prediction], self._raw_data.pos[has_prediction], self._raw_data.pos, k=1,
        )
        self._full_res_preds = full_pred

    def __repr__(self):
        return "{}(num_pos={})".format(self.__class__.__name__, self._num_pos)


class SegmentationTracker(BaseTracker):
    def __init__(
        self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False, ignore_label: int = IGNORE_LABEL
    ):
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
        super(SegmentationTracker, self).__init__(stage, wandb_log, use_tensorboard)
        self._num_classes = dataset.num_classes
        self._ignore_label = ignore_label
        self._dataset = dataset
        self.reset(stage)

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._confusion_matrix = ConfusionMatrix(self._num_classes)
        self._acc = 0
        self._macc = 0
        self._miou = 0

    @staticmethod
    def detach_tensor(tensor):
        if torch.torch.is_tensor(tensor):
            tensor = tensor.detach()
        return tensor

    @property
    def confusion_matrix(self):
        return self._confusion_matrix.confusion_matrix

    def track(self, model: model_interface.TrackerInterface, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        if not self._dataset.has_labels(self._stage):
            return

        super().track(model)

        outputs = model.get_output()
        targets = model.get_labels()

        # Mask ignored label
        mask = targets != self._ignore_label
        outputs = outputs[mask]
        targets = targets[mask]

        outputs = self._convert(outputs)
        targets = self._convert(targets)

        if len(targets) == 0:
            return

        assert outputs.shape[0] == len(targets)
        self._confusion_matrix.count_predicted_batch(targets, np.argmax(outputs, 1))

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

    @property
    def metric_func(self):
        self._metric_func = {
            "miou": max,
            "macc": max,
            "acc": max,
            "loss": min,
            "map": max,
        }  # Those map subsentences to their optimization functions
        return self._metric_func
