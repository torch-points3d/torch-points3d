from typing import Dict, List, Any
import torchnet as tnt
import torch
from collections import OrderedDict

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
        self._metric_func = {"loss": min, "acc": max, "pos": max, "neg": min, "map": max}

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._pred_boxes: Dict[str, List[BoxData]] = {}
        self._gt_boxes: Dict[str, List[BoxData]] = {}
        self._rec: Dict[str, Dict[str, float]] = {}
        self._ap: Dict[str, Dict[str, float]] = {}
        self._neg_ratio = tnt.meter.AverageValueMeter()
        self._obj_acc = tnt.meter.AverageValueMeter()
        self._pos_ratio = tnt.meter.AverageValueMeter()

    @staticmethod
    def detach_tensor(tensor):
        if torch.torch.is_tensor(tensor):
            tensor = tensor.detach()
        return tensor

    def track(self, model: TrackerInterface, data=None, track_boxes=False, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        if tracking boxes, you must provide a labeled "data" object with the following attributes:
            - id_scan: id of the scan to which the boxes belong to
            - instance_box_cornerimport torchnet as tnts - gt box corners
            - box_label_mask - mask for boxes (0 = no box)
            - sem_cls_label - semantic label for each box
        """
        super().track(model)

        outputs: VoteNetResults = model.get_output()

        total_num_proposal = outputs.objectness_label.shape[0] * outputs.objectness_label.shape[1]
        pos_ratio = torch.sum(outputs.objectness_label.float()).item() / float(total_num_proposal)
        self._pos_ratio.add(pos_ratio)
        self._neg_ratio.add(torch.sum(outputs.objectness_mask.float()).item() / float(total_num_proposal) - pos_ratio)

        obj_pred_val = torch.argmax(outputs.objectness_scores, 2)  # B,K
        self._obj_acc.add(
            torch.sum((obj_pred_val == outputs.objectness_label.long()).float() * outputs.objectness_mask).item()
            / (torch.sum(outputs.objectness_mask) + 1e-6).item()
        )

        if data is None or self._stage == "train" or not track_boxes:
            return

        self._add_box_pred(outputs, data, model.conv_type)

    def _add_box_pred(self, outputs: VoteNetResults, input_data, conv_type):
        # Track box predictions
        pred_boxes = outputs.get_boxes(self._dataset, apply_nms=True, duplicate_boxes=False)
        if input_data.id_scan is None:
            raise ValueError("Cannot track boxes without knowing in which scan they are")

        scan_ids = input_data.id_scan
        assert len(scan_ids) == len(pred_boxes)
        for idx, scan_id in enumerate(scan_ids):
            # Predictions
            self._pred_boxes[scan_id.item()] = pred_boxes[idx]

            # Ground truth
            sample_mask = idx
            gt_boxes = input_data.instance_box_corners[sample_mask]
            gt_boxes = gt_boxes[input_data.box_label_mask[sample_mask]]
            sample_labels = input_data.sem_cls_label[sample_mask]
            gt_box_data = [BoxData(sample_labels[i].item(), gt_boxes[i]) for i in range(len(gt_boxes))]
            self._gt_boxes[scan_id.item()] = gt_box_data

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        metrics["{}_acc".format(self._stage)] = meter_value(self._obj_acc)
        metrics["{}_pos".format(self._stage)] = meter_value(self._pos_ratio)
        metrics["{}_neg".format(self._stage)] = meter_value(self._neg_ratio)

        if self._has_box_data:
            for thresh, ap in self._ap.items():
                mAP = sum(ap.values()) / len(ap)
                metrics["{}_map{}".format(self._stage, thresh)] = mAP

        if verbose and self._has_box_data:
            for thresh in self._ap:
                metrics["{}_class_rec{}".format(self._stage, thresh)] = self._dict_to_str(self._rec[thresh])
                metrics["{}_class_ap{}".format(self._stage, thresh)] = self._dict_to_str(self._ap[thresh])

        return metrics

    def finalise(self, track_boxes=False, overlap_thresholds=[0.25, 0.5], **kwargs):
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

    @property
    def _has_box_data(self):
        return len(self._rec)

    @property
    def metric_func(self):
        return self._metric_func
