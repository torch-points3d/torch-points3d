from typing import Dict
import torchnet as tnt
import logging
from torch_geometric.nn.unpool import knn_interpolate

from src.models.base_model import BaseModel
from src.metrics.confusion_matrix import ConfusionMatrix
from src.metrics.segmentation_tracker import SegmentationTracker
from src.metrics.base_tracker import BaseTracker, meter_value
from src.datasets.segmentation import IGNORE_LABEL

log = logging.getLogger(__name__)


class S3DISTracker(SegmentationTracker):
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._test_area = None
        self._full_vote_miou = None
        self._vote_miou = None

    def track(self, model: BaseModel, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)
        # Train mode, nothing special to do
        if self._stage == "train":
            return

        # Test mode, compute votes in order to get full res predictions
        if self._test_area is None:
            self._test_area = self._dataset.test_data.clone()
            assert self._test_area.y is not None
            self._test_area.has_prediction = torch.zeros(self._test_area.y.shape[0], dtype=torch.bool)
            self._test_area.votes = torch.zeros((self._test_area.y.shape[0], self._num_classes), dtype=torch.float)

        # Gather input to the model and check that it fits with the test set
        inputs = model.get_input()
        assert inputs.originid is not None
        assert inputs.originid.max()[0] < self._test_area.pos.shape[0]

        # Set predictions
        outputs = model.get_output()
        self._test_area.votes[inputs.originid] += outputs
        self._test_area.has_prediction[inputs.originid] = True

    def _compute_full_miou(self):
        if self._full_vote_miou is not None:
            return

        log.debug(
            "Computing full res mIoU, we have predictions for %i%% of the points."
            % torch.sum(self._test_area.has_prediction)
            / (1.0 * self._test_area.has_prediction.shape[0])
        )

        # Complete for points that have a prediction
        c = ConfusionMatrix(self._num_classes)
        gt = self._test_area.y[self._test_area.has_prediction].cpu().numpy()
        pred = torch.max(self._test_area.votes[self._test_area.has_prediction], 1)[1].cpu().numpy()
        c.count_predicted_batch(gt, pred)
        self._vote_miou = c.get_average_intersection_union()

        # Full res interpolation
        full_pred = knn_interpolate(
            self._test_area.pos[self._test_area.has_prediction],
            self._test_area.votes[self._test_area.has_prediction],
            self._test_area.pos,
            k=1,
        )
        c = ConfusionMatrix(self._num_classes)
        c.count_predicted_batch(self._test_area.y.cpu().numpy(), full_pred.cpu().numpy())
        self._full_vote_miou = c.get_average_intersection_union()

    def get_metrics(self, verbose=False) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        if verbose and self._stage != "train":
            self._compute_full_miou()
            metrics["{}_full_vote_miou".format(self._stage)] = self._full_vote_miou
            metrics["{}_vote_miou".format(self._stage)] = self._vote_miou
        return metrics
