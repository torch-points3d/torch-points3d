from typing import Dict
import torchnet as tnt
import logging
import torch
import time
from torch_geometric.nn.unpool import knn_interpolate

from torch_points3d.models.base_model import BaseModel
from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.core.data_transform import SaveOriginalPosId

log = logging.getLogger(__name__)


class S3DISTracker(SegmentationTracker):
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._test_area = None
        self._full_vote_miou = None
        self._vote_miou = None

    def track(self, model: BaseModel, full_res=False, **kwargs):
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
            self._test_area.has_prediction = torch.zeros(self._test_area.y.shape[0], dtype=torch.bool)
            self._test_area.votes = torch.zeros((self._test_area.y.shape[0], self._num_classes), dtype=torch.float)
            self._test_area.to(model.device)

        # Gather input to the model and check that it fits with the test set
        inputs = model.get_input()
        if inputs[SaveOriginalPosId.KEY] is None or inputs[SaveOriginalPosId.KEY].max() >= self._test_area.pos.shape[0]:
            raise ValueError(
                "The inputs given to the model do not have a %s attribute or this attribute does\
                     not correspond to the number of points in the test area point cloud."
                % SaveOriginalPosId.KEY
            )

        # Set predictions
        outputs = model.get_output()
        self._test_area.votes[inputs[SaveOriginalPosId.KEY]] += outputs
        self._test_area.has_prediction[inputs[SaveOriginalPosId.KEY]] = True

    def finalise(self, full_res=False, **kwargs):
        if full_res:
            self._compute_full_miou()

    def _compute_full_miou(self):
        if self._full_vote_miou is not None:
            return

        self._dataset.to_ply(
            self._test_area.pos[self._test_area.has_prediction].cpu(),
            torch.argmax(self._test_area.votes[self._test_area.has_prediction], 1).cpu().numpy(),
            "test.ply",
        )

        log.info(
            "Computing full res mIoU, we have predictions for %.2f%% of the points."
            % (torch.sum(self._test_area.has_prediction) / (1.0 * self._test_area.has_prediction.shape[0]) * 100)
        )

        self._test_area = self._test_area.to("cpu")

        # Complete for points that have a prediction
        print("Start confusion matrix")
        t = time.time()
        c = ConfusionMatrix(self._num_classes)
        gt = self._test_area.y[self._test_area.has_prediction].numpy()
        pred = torch.argmax(self._test_area.votes[self._test_area.has_prediction], 1).numpy()
        c.count_predicted_batch(gt, pred)
        self._vote_miou = c.get_average_intersection_union() * 100
        per_class_iou = c.get_intersection_union_per_class()[0]
        per_class_iou = {self._dataset.INV_OBJECT_LABEL[k]: v for k, v in enumerate(per_class_iou)}
        print(per_class_iou)
        print("Low res timing: %.2f" % (time.time() - t))

        # Full res interpolation
        t = time.time()
        full_pred = knn_interpolate(
            self._test_area.votes[self._test_area.has_prediction],
            self._test_area.pos[self._test_area.has_prediction],
            self._test_area.pos,
            k=1,
        )
        print("knn timing: %.2f" % (time.time() - t))

        # Full res pred
        t = time.time()
        c = ConfusionMatrix(self._num_classes)
        c.count_predicted_batch(self._test_area.y.numpy(), torch.argmax(full_pred, 1).numpy())
        self._full_vote_miou = c.get_average_intersection_union() * 100
        print("Full res timing: %.2f" % (time.time() - t))

    def get_metrics(self, verbose=False) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        if verbose and self._vote_miou:
            metrics["{}_full_vote_miou".format(self._stage)] = self._full_vote_miou
            metrics["{}_vote_miou".format(self._stage)] = self._vote_miou
        return metrics
