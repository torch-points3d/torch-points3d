import torchnet as tnt

from torch_points3d.models.model_interface import TrackerInterface
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.metrics.base_tracker import BaseTracker
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.models.panoptic.structures import PanopticResults, PanopticLabels


class PanopticTracker(SegmentationTracker):
    """ Class that provides tracking of semantic segmentation as well as
    instance segmentation """

    def track(self, model: TrackerInterface, **kwargs):
        """ Track metrics for panoptic segmentation
        """
        BaseTracker.track(self, model)
        outputs: PanopticResults = model.get_output()
        labels: PanopticLabels = model.get_labels()

        # Track semantic
        super()._compute_metrics(outputs.semantic_logits, labels.y)

        # Track instance
        # TODO
