from typing import Dict, List, Any
import torchnet as tnt
import torch

from torch_points3d.models.model_interface import TrackerInterface
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.metrics.meters import APMeter
from torch_points3d.datasets.segmentation import IGNORE_LABEL

from torch_points3d.modules.VoteNet import VoteNetResults
from torch_points3d.datasets.object_detection.box_data import BoxData
from .box_detection.ap import eval_detection


class ObjectDetectionTracker(BaseTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False):
        super(ObjectDetectionTracker, self).__init__(stage, wandb_log, use_tensorboard)
        self._num_classes = dataset.num_classes
        self._dataset = dataset
        self.reset(stage)
        self._metric_func = {"loss": min, "acc": max}
        self._pred_boxes: Dict[str, List[BoxData]] = {}
        self._gt_boxes: Dict[str, List[BoxData]] = {}
        self._rec: Dict[str, float] = {}
        self._ap: Dict[str, float] = {}

    def reset(self, stage="train"):
        super().reset(stage=stage)

    @staticmethod
    def detach_tensor(tensor):
        if torch.torch.is_tensor(tensor):
            tensor = tensor.detach()
        return tensor

    def track(self, model: TrackerInterface, data=None, track_boxes=False, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        if tracking boxes, you must provide a labeled "data" object with the following attributes:
            - id_scan: id of the scan to which the boxes belong to
            - instance_box_corners - gt box corners
            - box_label_mask - mask for boxes (0 = no box)
            - sem_cls_label - semantic label for each box
        """
        super().track(model)

        outputs: VoteNetResults = model.get_output()

        total_num_proposal = outputs.objectness_label.shape[0] * outputs.objectness_label.shape[1]
        self._pos_ratio = torch.sum(outputs.objectness_label.float()) / float(total_num_proposal)
        self._neg_ratio = torch.sum(outputs.objectness_label.float()) / float(total_num_proposal) - self._pos_ratio

        obj_pred_val = torch.argmax(outputs.objectness_scores, 2)  # B,K
        self._obj_acc = torch.sum(
            (obj_pred_val == outputs.objectness_label.long()).float() * outputs.objectness_mask
        ) / (torch.sum(outputs.objectness_mask) + 1e-6)

        if data is None or self._stage == "train" or not track_boxes:
            return

    def _add_box_pred(self, outputs: VoteNetResults, input_data, conv_type):
        # Track box predictions
        pred_boxes = outputs.get_boxes(self._dataset, apply_nms=True)
        if input_data.id_scan is None:
            raise ValueError("Cannot track boxes without knowing in which scan they are")

        scan_ids = input_data.id_scan
        assert len(scan_ids) == len(pred_boxes)
        for idx, scan_id in enumerate(scan_ids):
            # Predictions
            self._pred_boxes[scan_id.item()] = pred_boxes[idx]

            # Ground truth
            sample_mask = idx
            if conv_type != "DENSE":
                sample_mask = input_data.batch == idx
            gt_boxes = input_data.instance_box_corners[sample_mask]
            gt_boxes = gt_boxes[input_data.box_label_mask[sample_mask]]
            sample_labels = input_data.sem_cls_label[sample_mask]
            gt_box_data = [BoxData(sample_labels[i].item(), gt_boxes[i]) for i in range(len(gt_boxes))]
            self._gt_boxes[scan_id.item()] = gt_box_data

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        metrics["{}_acc".format(self._stage)] = self._obj_acc.item()
        metrics["{}_pos".format(self._stage)] = self._pos_ratio.item()
        metrics["{}_neg".format(self._stage)] = self._neg_ratio.item()

        if self._has_box_data:
            mAP = sum(self._ap.values()) / len(self._ap)
            metrics["{}_map".format(self._stage)] = mAP

        if verbose and self._has_box_data:
            metrics["{}_class_rec".format(self._stage)] = self._rec
            metrics["{}_class_ap".format(self._stage)] = self._ap

        return metrics

    def finalise(self, track_boxes=False, overlap_threshold=0.25, **kwargs):
        if not track_boxes or len(self._gt_boxes) == 0:
            return

        # Compute box detection metrics
        self._rec, _, self._ap = eval_detection(self._pred_boxes, self._gt_boxes, ovthresh=overlap_threshold)

    @property
    def _has_box_data(self):
        return len(self._rec)

    @property
    def metric_func(self):
        return self._metric_func
