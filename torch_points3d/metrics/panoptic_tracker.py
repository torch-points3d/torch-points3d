import torchnet as tnt

from torch_points3d.models.model_interface import TrackerInterface
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.metrics.base_tracker import BaseTracker
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.models.panoptic.structures import PanopticResults, PanopticLabels
from torch_points_kernels import instance_iou


class PanopticTracker(SegmentationTracker):
    """ Class that provides tracking of semantic segmentation as well as
    instance segmentation """

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._obj_acc = tnt.meter.AverageValueMeter()

    def track(self, model: TrackerInterface, data=None, **kwargs):
        """ Track metrics for panoptic segmentation
        """
        BaseTracker.track(self, model)
        outputs: PanopticResults = model.get_output()
        labels: PanopticLabels = model.get_labels()

        # Track semantic
        super()._compute_metrics(outputs.semantic_logits, labels.y)

        if not data:
            return

        # Object accuracy
        valid_cluster_idx = outputs.get_instances()
        clusters = [outputs.clusters[i] for i in valid_cluster_idx]
        instance_iou(clusters, labels.instance_labels, data.batch).max(1)
