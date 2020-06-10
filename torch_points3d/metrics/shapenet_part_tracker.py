import numpy as np
from .confusion_matrix import ConfusionMatrix
from .base_tracker import meter_value, BaseTracker
from torch_geometric.data import Data

from torch_points3d.models import model_interface
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.metrics.segmentation_helpers import SegmentationVoter


class ShapenetPartTracker(BaseTracker):
    def __init__(self, dataset, stage: str = "train", wandb_log: bool = False, use_tensorboard: bool = False):
        """ Segmentation tracker shapenet part seg problem. The dataset needs to have a
        class_to_segment member that defines how metrics get computed and agregated.
        It follows shapenet official formula for computing miou which treats missing part as having an iou of 1
        See https://github.com/charlesq34/pointnet2/blob/42926632a3c33461aebfbee2d829098b30a23aaa/part_seg/evaluate.py#L166-L176

        Arguments:
            dataset {[type]}

        Keyword Arguments:
            stage {str} -- current stage (default: {"train"})
            wandb_log {bool} -- Log to Wanndb (default: {False})
            use_tensorboard {bool} -- Log to tensorboard (default: {False})
        """
        super(ShapenetPartTracker, self).__init__(stage, wandb_log, use_tensorboard)
        self._dataset = dataset
        self._num_classes = dataset.num_classes
        self._class_seg_map = dataset.class_to_segments
        self._seg_to_class = {}
        for cat, segments in self._class_seg_map.items():
            for label in segments:
                self._seg_to_class[label] = cat
        self.reset(stage=stage)

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._shape_ious = {cat: [] for cat in self._class_seg_map.keys()}
        self._Cmiou = 0
        self._Imiou = 0
        self._miou_per_class = {}
        self._full_res_scans = {cat: {} for cat in self._class_seg_map.keys()}
        self._full_shape_ious = {cat: [] for cat in self._class_seg_map.keys()}
        self._full_miou_per_class = None
        self._full_Cmiou = None
        self._full_Imiou = None
        self._full_res = False

    def track(self, model: model_interface.TrackerInterface, full_res: bool = False, data: Data = None, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)
        self._conv_type = model.conv_type
        outputs = self._convert(model.get_output())
        targets = self._convert(model.get_labels())
        batch_idx = self._convert(model.get_batch())
        if batch_idx is None:
            raise ValueError("Your model need to set the batch_idx variable in its set_input function.")

        nb_batches = batch_idx.max() + 1

        if self._stage != "train" and full_res:
            self._add_votes(data, outputs, batch_idx)

        # pred to the groundtruth classes (selected by seg_classes[cat])
        for b in range(nb_batches):
            segl = targets[batch_idx == b]
            cat = self._seg_to_class[segl[0]]
            logits = outputs[batch_idx == b, :]  # (num_points, num_classes)
            segp = logits[:, self._class_seg_map[cat]].argmax(1) + self._class_seg_map[cat][0]
            part_ious = self._compute_part_ious(segl, segp, cat)
            self._shape_ious[cat].append(np.mean(part_ious))

        self._miou_per_class, self._Cmiou, self._Imiou = ShapenetPartTracker._get_metrics_per_class(self._shape_ious)

    def _add_votes(self, data, outputs, batch_idx):
        nb_batches = batch_idx.max() + 1
        for b in range(nb_batches):
            batch_mask = b
            if self._conv_type != "DENSE":
                batch_mask = batch_idx == b
            segl = data.y[batch_mask][0].item()

            cat = self._seg_to_class[segl]
            logits = outputs[batch_idx == b, :]  # (num_points, num_classes)

            id_scan = data.id_scan[b].item()
            if id_scan not in self._full_res_scans[cat]:
                raw_data = self._dataset.get_raw_data(self._stage, id_scan)
                self._full_res_scans[cat][id_scan] = SegmentationVoter(
                    raw_data, self._num_classes, self._conv_type, class_seg_map=self._class_seg_map[cat]
                )
            self._full_res_scans[cat][id_scan].add_vote(data, logits, batch_mask)

    def finalise(self, **kwargs):
        # Check if at least one element has been created for full res interpolation
        contains_elements = np.sum([bool(d) for d in list(self._full_res_scans.values())]) > 0
        if not contains_elements:
            return

        for cat in self._full_res_scans.keys():
            samples = self._full_res_scans[cat].values()
            for sample in samples:
                segl = sample.full_res_labels.numpy()
                segp = sample.full_res_preds.numpy()
                part_ious = self._compute_part_ious(segl, segp, cat)
                self._full_shape_ious[cat].append(np.mean(part_ious))
        self._full_miou_per_class, self._full_Cmiou, self._full_Imiou = ShapenetPartTracker._get_metrics_per_class(
            self._full_shape_ious
        )

        self._full_res = True

    def _compute_part_ious(self, segl, segp, cat):
        part_ious = np.zeros(len(self._class_seg_map[cat]))
        for l in self._class_seg_map[cat]:
            if np.sum((segl == l) | (segp == l)) == 0:
                # part is not present in this shape
                part_ious[l - self._class_seg_map[cat][0]] = 1
            else:
                part_ious[l - self._class_seg_map[cat][0]] = float(np.sum((segl == l) & (segp == l))) / float(
                    np.sum((segl == l) | (segp == l))
                )
        return part_ious

    def get_metrics(self, verbose=False):
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)
        metrics["{}_Cmiou".format(self._stage)] = self._Cmiou * 100
        metrics["{}_Imiou".format(self._stage)] = self._Imiou * 100
        if self._full_res:
            metrics["{}_full_Cmiou".format(self._stage)] = self._full_Cmiou * 100
            metrics["{}_full_Imiou".format(self._stage)] = self._full_Imiou * 100
        if verbose:
            metrics["{}_Imiou_per_class".format(self._stage)] = self._miou_per_class
            if self._full_res:
                metrics["{}_full_Imiou_per_class".format(self._stage)] = self._full_miou_per_class
        return metrics

    @property
    def metric_func(self):
        self._metric_func = {"Cmiou": max, "Imiou": max, "loss": min}
        return self._metric_func

    @staticmethod
    def _get_metrics_per_class(shape_ious):
        instance_ious = []
        cat_ious = {}
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                instance_ious.append(iou)
            if len(shape_ious[cat]):
                cat_ious[cat] = np.mean(shape_ious[cat])
        mean_class_ious = np.mean(list(cat_ious.values()))
        return cat_ious, mean_class_ious, np.mean(instance_ious)
