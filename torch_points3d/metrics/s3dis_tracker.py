from typing import Dict, Any
import logging
import torch
from torch_geometric.nn.unpool import knn_interpolate

from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.models import model_interface

log = logging.getLogger(__name__)


class S3DISTracker(SegmentationTracker):
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._test_area = None
        self._full_vote_miou = None
        self._vote_miou = None
        self._full_confusion = None
        self._iou_per_class = {}

    def track(self, model: model_interface.TrackerInterface, full_res=False, data=None, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)

        # Train mode or low res, nothing special to do
        if self._stage == "train" or not full_res:
            return

        # Test mode, compute votes in order to get full res predictions
        if self._test_area is None:
            self._test_area = self._dataset.test_data.clone()
            if self._test_area.y is None:
                raise ValueError("It seems that the test area data does not have labels (attribute y).")
            self._test_area.prediction_count = torch.zeros(self._test_area.y.shape[0], dtype=torch.int)
            self._test_area.votes = torch.zeros((self._test_area.y.shape[0], self._num_classes), dtype=torch.float)
            self._test_area.to(model.device)

        # Gather origin ids and check that it fits with the test set
        inputs = data if data is not None else model.get_input()
        if inputs[SaveOriginalPosId.KEY] is None:
            raise ValueError("The inputs given to the model do not have a %s attribute." % SaveOriginalPosId.KEY)

        originids = inputs[SaveOriginalPosId.KEY]
        if originids.dim() == 2:
            originids = originids.flatten()
        if originids.max() >= self._test_area.pos.shape[0]:
            raise ValueError("Origin ids are larger than the number of points in the original point cloud.")

        # Set predictions
        outputs = model.get_output()
        self._test_area.votes[originids] += outputs
        self._test_area.prediction_count[originids] += 1

    def finalise(self, full_res=False, vote_miou=True, ply_output="", **kwargs):
        per_class_iou = self._confusion_matrix.get_intersection_union_per_class()[0]
        self._iou_per_class = {self._dataset.INV_OBJECT_LABEL[k]: v for k, v in enumerate(per_class_iou)}

        if vote_miou and self._test_area:
            # Complete for points that have a prediction
            self._test_area = self._test_area.to("cpu")
            c = ConfusionMatrix(self._num_classes)
            has_prediction = self._test_area.prediction_count > 0
            gt = self._test_area.y[has_prediction].numpy()
            pred = torch.argmax(self._test_area.votes[has_prediction], 1).numpy()
            c.count_predicted_batch(gt, pred)
            self._vote_miou = c.get_average_intersection_union() * 100

        if full_res:
            self._compute_full_miou()

        if ply_output:
            has_prediction = self._test_area.prediction_count > 0
            self._dataset.to_ply(
                self._test_area.pos[has_prediction].cpu(),
                torch.argmax(self._test_area.votes[has_prediction], 1).cpu().numpy(),
                ply_output,
            )

    def _compute_full_miou(self):
        if self._full_vote_miou is not None:
            return

        has_prediction = self._test_area.prediction_count > 0
        log.info(
            "Computing full res mIoU, we have predictions for %.2f%% of the points."
            % (torch.sum(has_prediction) / (1.0 * has_prediction.shape[0]) * 100)
        )

        self._test_area = self._test_area.to("cpu")

        # Full res interpolation
        full_pred = knn_interpolate(
            self._test_area.votes[has_prediction], self._test_area.pos[has_prediction], self._test_area.pos, k=1,
        )

        # Full res pred
        self._full_confusion = ConfusionMatrix(self._num_classes)
        self._full_confusion.count_predicted_batch(self._test_area.y.numpy(), torch.argmax(full_pred, 1).numpy())
        self._full_vote_miou = self._full_confusion.get_average_intersection_union() * 100

    @property
    def full_confusion_matrix(self):
        return self._full_confusion

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        if verbose:
            metrics["{}_iou_per_class".format(self._stage)] = self._iou_per_class
            if self._vote_miou:
                metrics["{}_full_vote_miou".format(self._stage)] = self._full_vote_miou
                metrics["{}_vote_miou".format(self._stage)] = self._vote_miou
        return metrics
