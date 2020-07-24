import torchnet as tnt
from typing import NamedTuple, Dict, Any, List, Tuple
import torch
import numpy as np
from torch_scatter import scatter_add
from collections import OrderedDict, defaultdict

from torch_points3d.models.model_interface import TrackerInterface
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.models.panoptic.structures import PanopticResults, PanopticLabels
from torch_points_kernels import instance_iou
from .box_detection.ap import voc_ap


class _Instance(NamedTuple):
    classname: str
    score: float
    indices: np.array
    scan_id: int

    def iou(self, other: "_Instance") -> float:
        assert self.scan_id == other.scan_id
        intersection = float(len(np.intersect1d(other.indices, self.indices)))
        return intersection / float(len(other.indices) + len(self.indices) - intersection)

    def find_best_match(self, others: List["_Instance"]) -> Tuple[float, int]:
        ioumax = -np.inf
        best_match = -1
        for i, other in enumerate(others):
            iou = self.iou(other)
            if iou > ioumax:
                ioumax = iou
                best_match = i
        return ioumax, best_match


class InstanceAPMeter:
    def __init__(self):
        self._pred_clusters = defaultdict(list)  # {classname: List[_Instance]}
        self._gt_clusters = defaultdict(lambda: defaultdict(list))  # {classname:{scan_id: List[_Instance]}

    def add(self, pred_clusters: List[_Instance], gt_clusters: List[_Instance]):
        for instance in pred_clusters:
            self._pred_clusters[instance.classname].append(instance)
        for instance in gt_clusters:
            self._gt_clusters[instance.classname][instance.scan_id].append(instance)

    def _eval_cls(self, classname, iou_threshold):
        preds = self._pred_clusters.get(classname, [])
        allgts = self._gt_clusters.get(classname, {})
        visited = {scan_id: len(gt) * [False] for scan_id, gt in allgts.items()}
        ngt = 0
        for gts in allgts.values():
            ngt += len(gts)

        # Start with most confident first
        preds.sort(key=lambda x: x.score, reverse=True)
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        for p, pred in enumerate(preds):
            scan_id = pred.scan_id
            gts = allgts.get(scan_id, [])
            if len(gts) == 0:
                fp[p] = 1
                continue

            # Find best macth in ground truth
            ioumax, best_match = pred.find_best_match(gts)

            if ioumax < iou_threshold:
                fp[p] = 1
                continue

            if visited[scan_id][best_match]:
                fp[p] = 1
            else:
                visited[scan_id][best_match] = True
                tp[p] = 1

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(ngt)

        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec)
        return rec, prec, ap

    def eval(self, iou_threshold, processes=1):
        rec = {}
        prec = {}
        ap = {}
        for classname in self._gt_clusters.keys():
            rec[classname], prec[classname], ap[classname] = self._eval_cls(classname, iou_threshold)

        for i, classname in enumerate(self._gt_clusters.keys()):
            if classname not in self._pred_clusters:
                rec[classname] = 0
                prec[classname] = 0
                ap[classname] = 0

        return rec, prec, ap


class PanopticTracker(SegmentationTracker):
    """ Class that provides tracking of semantic segmentation as well as
    instance segmentation """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metric_func = {**self._metric_func, "pos": max, "neg": min, "map": max}

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._pos = tnt.meter.AverageValueMeter()
        self._neg = tnt.meter.AverageValueMeter()
        self._acc_meter = tnt.meter.AverageValueMeter()
        self._ap_meter = InstanceAPMeter()
        self._scan_id_offset = 0
        self._rec: Dict[str, float] = {}
        self._ap: Dict[str, float] = {}

    def track(
        self,
        model: TrackerInterface,
        data=None,
        iou_threshold=0.25,
        track_instances=True,
        min_cluster_points=10,
        **kwargs
    ):
        """ Track metrics for panoptic segmentation
        """
        self._iou_threshold = iou_threshold
        BaseTracker.track(self, model)
        outputs: PanopticResults = model.get_output()
        labels: PanopticLabels = model.get_labels()

        # Track semantic
        super()._compute_metrics(outputs.semantic_logits, labels.y)

        if not data:
            return
        assert data.pos.dim() == 2, "Only supports packed batches"

        # Object accuracy
        clusters = PanopticTracker._extract_clusters(outputs, min_cluster_points)
        if not clusters:
            return

        predicted_labels = outputs.semantic_logits.max(1)[1]
        tp, fp, acc = self._compute_acc(
            clusters, predicted_labels, labels, data.batch, labels.num_instances, iou_threshold
        )
        self._pos.add(tp)
        self._neg.add(fp)
        self._acc_meter.add(acc)

        # Track instances for AP
        if track_instances:
            pred_clusters = self._pred_instances_per_scan(
                clusters, predicted_labels, outputs.cluster_scores, data.batch, self._scan_id_offset
            )
            gt_clusters = self._gt_instances_per_scan(
                labels.instance_labels, labels.y, data.batch, self._scan_id_offset
            )
            self._ap_meter.add(pred_clusters, gt_clusters)
            self._scan_id_offset += data.batch[-1].item() + 1

    def finalise(self, track_instances=True, **kwargs):
        if not track_instances:
            return

        rec, _, ap = self._ap_meter.eval(self._iou_threshold)
        self._ap = OrderedDict(sorted(ap.items()))
        self._rec = OrderedDict({})
        for key, val in sorted(rec.items()):
            try:
                value = val[-1]
            except TypeError:
                value = val
            self._rec[key] = value

    @staticmethod
    def _compute_acc(clusters, predicted_labels, labels, batch, num_instances, iou_threshold):
        """ Computes the ratio of True positives, False positives and accuracy
        """
        iou_values, gt_ids = instance_iou(clusters, labels.instance_labels, batch).max(1)
        gt_ids += 1
        instance_offsets = torch.cat((torch.tensor([0]).to(num_instances.device), num_instances.cumsum(-1)))
        tp = 0
        fp = 0
        for i, iou in enumerate(iou_values):
            # Too low iou, no match in ground truth
            if iou < iou_threshold:
                fp += 1
                continue

            # Check that semantic is correct
            sample_idx = batch[clusters[i][0]]
            sample_mask = batch == sample_idx
            instance_offset = instance_offsets[sample_idx]
            gt_mask = labels.instance_labels[sample_mask] == (gt_ids[i] - instance_offset)
            gt_classes = labels.y[sample_mask][torch.nonzero(gt_mask, as_tuple=False)]
            gt_classes, counts = torch.unique(gt_classes, return_counts=True)
            gt_class = gt_classes[counts.max(-1)[1]]
            pred_class = predicted_labels[clusters[i][0]]
            if gt_class == pred_class:
                tp += 1
            else:
                fp += 1
        acc = tp / len(clusters)
        tp = tp / torch.sum(labels.num_instances).cpu().item()
        fp = fp / torch.sum(labels.num_instances).cpu().item()
        return tp, fp, acc

    @staticmethod
    def _extract_clusters(outputs, min_cluster_points):
        valid_cluster_idx = outputs.get_instances(min_cluster_points=min_cluster_points)
        clusters = [outputs.clusters[i] for i in valid_cluster_idx]
        return clusters

    @staticmethod
    def _pred_instances_per_scan(clusters, predicted_labels, scores, batch, scan_id_offset):
        # Get sample index offset
        ones = torch.ones_like(batch)
        sample_sizes = torch.cat((torch.tensor([0]).to(batch.device), scatter_add(ones, batch)))
        offsets = sample_sizes.cumsum(dim=-1).cpu().numpy()

        # Build instance objects
        instances = []
        for i, cl in enumerate(clusters):
            sample_idx = batch[cl[0]].item()
            scan_id = sample_idx + scan_id_offset
            indices = cl.cpu().numpy() - offsets[sample_idx]
            instances.append(
                _Instance(
                    classname=predicted_labels[cl[0]].item(), score=scores[i].item(), indices=indices, scan_id=scan_id
                )
            )
        return instances

    @staticmethod
    def _gt_instances_per_scan(instance_labels, gt_labels, batch, scan_id_offset):
        batch_size = batch[-1] + 1
        instances = []
        for b in range(batch_size):
            sample_mask = batch == b
            instances_in_sample = instance_labels[sample_mask]
            gt_labels_sample = gt_labels[sample_mask]
            num_instances = torch.max(instances_in_sample)
            scan_id = b + scan_id_offset
            for i in range(num_instances):
                instance_indices = torch.where(instances_in_sample == i + 1)[0].cpu().numpy()
                instances.append(
                    _Instance(
                        classname=gt_labels_sample[instance_indices[0]].item(),
                        score=-1,
                        indices=instance_indices,
                        scan_id=scan_id,
                    )
                )
        return instances

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        metrics["{}_pos".format(self._stage)] = meter_value(self._pos)
        metrics["{}_neg".format(self._stage)] = meter_value(self._neg)
        metrics["{}_Iacc".format(self._stage)] = meter_value(self._acc_meter)

        if self._has_instance_data:
            mAP = sum(self._ap.values()) / len(self._ap)
            metrics["{}_map".format(self._stage)] = mAP

        if verbose and self._has_instance_data:
            metrics["{}_class_rec".format(self._stage)] = self._dict_to_str(self._rec)
            metrics["{}_class_ap".format(self._stage)] = self._dict_to_str(self._ap)
        return metrics

    @property
    def _has_instance_data(self):
        return len(self._rec)
