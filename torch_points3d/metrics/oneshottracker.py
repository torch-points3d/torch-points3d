from typing import Any, Dict

from torch_points3d.models.model_interface import TrackerInterface
from torch_points3d.metrics.base_tracker import BaseTracker
from torch_points3d.datasets.object_detection.box_data import BoxData
from torch_points3d.utils.box_utils import box3d_iou


class OneShotObjectTracker(BaseTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False):
        super().__init__(stage, wandb_log, use_tensorboard)
        self.reset(stage)

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._tp = 0
        self._fp = 0
        self._ngt = 0

    def track(self, model: TrackerInterface, data=None, threshold=0.25, **kwargs):
        super().track(model)

        outputs = model.get_output()
        pred_boxes = outputs.get_boxes()

        if data:
            self._reshape_batch(data)
            scan_ids = data.id_scan
            assert len(scan_ids) == len(pred_boxes)
            for idx in range(len(scan_ids)):
                # Predictions
                sample_pred_boxes = pred_boxes[idx]

                # Ground truth
                sample_mask = idx
                gt_boxes = data.instance_box_corners[sample_mask]
                gt_boxes = gt_boxes[data.box_label_mask[sample_mask]]
                sample_labels = data.sem_cls_label[sample_mask]

                # No box found
                for i in range(len(gt_boxes)):
                    if sample_labels[i].item() == model.class_label:
                        self._ngt += 1

                # Found some boxes
                for pred_box in sample_pred_boxes:
                    found = False
                    for i in range(len(gt_boxes)):
                        if sample_labels[i].item() == pred_box.classname:
                            iou = box3d_iou(pred_box.corners3d, gt_boxes[i])
                            if iou > threshold:
                                if found:  # found a box that was already found
                                    self._fp += 1
                                else:
                                    self._tp += 1  # found an actual valid box
                                    found = True
                    if not found:  # found the wrong box
                        self._fp += 1

    def _reshape_batch(self, data):
        """ Ensures that the label tensors are unwrapped in case data comes from a sparse batch """
        batch_size = len(data.id_scan)
        if data.instance_box_corners.dim() == 3:
            data.instance_box_corners = data.instance_box_corners.reshape(batch_size, -1, 8, 3)
            data.box_label_mask = data.box_label_mask.reshape(batch_size, -1)
            data.sem_cls_label = data.sem_cls_label.reshape(batch_size, -1)
        assert data.instance_box_corners.dim() == 4
        assert data.box_label_mask.dim() == 2
        assert data.sem_cls_label.dim() == 2

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)
        metrics["{}_prec".format(self._stage)] = self._tp / max(1, (self._fp + self._tp))
        metrics["{}_rec".format(self._stage)] = self._tp / max(1, (self._ngt))

        return metrics
