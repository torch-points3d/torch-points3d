from typing import List, Dict
from collections import OrderedDict

from torch_points3d.models.model_interface import TrackerInterface
from torch_points3d.metrics.base_tracker import BaseTracker
from torch_points3d.datasets.object_detection.box_data import BoxData
from .box_detection.ap import eval_detection


class OneShotObjectTracker(BaseTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False):
        super().__init__(stage, wandb_log, use_tensorboard)
        self._oneshot_class = dataset.oneshot_class

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._pred_boxes: Dict[str, List[BoxData]] = {}
        self._gt_boxes: Dict[str, List[BoxData]] = {}

    def track(self, model: TrackerInterface, data=None, **kwargs):
        super().track(model)

        outputs = model.get_output()
        pred_boxes = outputs.get_boxes()

        if data:
            scan_ids = data.id_scan
            assert len(scan_ids) == len(pred_boxes)
            for idx, scan_id in enumerate(scan_ids):
                # Predictions
                self._pred_boxes[scan_id.item()] = pred_boxes[idx]

                # Ground truth
                sample_mask = idx
                gt_boxes = data.instance_box_corners[sample_mask]
                gt_boxes = gt_boxes[data.box_label_mask[sample_mask]]
                sample_labels = data.sem_cls_label[sample_mask]
                gt_box_data = []
                for i in range(len(gt_boxes)):
                    if sample_labels[i].item() == self._oneshot_class:
                        gt_box_data.append(BoxData(sample_labels[i].item(), gt_boxes[i]))
                self._gt_boxes[scan_id.item()] = gt_box_data

    def finalise(self, track_boxes=False, overlap_thresholds=[0.25], **kwargs):
        if not track_boxes or len(self._gt_boxes) == 0:
            return

        # Compute box detection metrics
        self._ap = {}
        self._rec = {}
        for thresh in overlap_thresholds:
            rec, _, ap = eval_detection(self._pred_boxes, self._gt_boxes, ovthresh=thresh)
            self._ap[str(thresh)] = OrderedDict(sorted(ap.items()))
            self._rec[str(thresh)] = OrderedDict({})
            for key, val in sorted(rec.items()):
                try:
                    value = val[-1]
                except TypeError:
                    value = val
                self._rec[str(thresh)][key] = value
