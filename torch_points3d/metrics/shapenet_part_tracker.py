from typing import Dict
import numpy as np

from .confusion_matrix import ConfusionMatrix
from .base_tracker import meter_value, BaseTracker
from torch_points3d.models import model_interface


class ShapenetPartTracker(BaseTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False):
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

    def track(self, model: model_interface.TrackerInterface, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)
        outputs = self._convert(model.get_output())
        targets = self._convert(model.get_labels())
        batch_idx = self._convert(model.get_batch())
        if batch_idx is None:
            raise ValueError("Your model need to set the batch_idx variable in its set_input function.")

        nb_batches = batch_idx.max() + 1

        # pred to the groundtruth classes (selected by seg_classes[cat])
        for b in range(nb_batches):
            segl = targets[batch_idx == b]
            cat = self._seg_to_class[segl[0]]
            logits = outputs[batch_idx == b, :]  # (num_points, num_classes)
            segp = logits[:, self._class_seg_map[cat]].argmax(1) + self._class_seg_map[cat][0]
            part_ious = np.zeros(len(self._class_seg_map[cat]))
            for l in self._class_seg_map[cat]:
                if np.sum((segl == l) | (segp == l)) == 0:
                    # part is not present in this shape
                    part_ious[l - self._class_seg_map[cat][0]] = 1
                else:
                    part_ious[l - self._class_seg_map[cat][0]] = float(np.sum((segl == l) & (segp == l))) / float(
                        np.sum((segl == l) | (segp == l))
                    )
            self._shape_ious[cat].append(np.mean(part_ious))

        self._miou_per_class, self._Cmiou, self._Imiou = self._get_metrics_per_class()

    def _get_metrics_per_class(self):
        instance_ious = []
        cat_ious = {}
        for cat in self._shape_ious.keys():
            for iou in self._shape_ious[cat]:
                instance_ious.append(iou)
            if len(self._shape_ious[cat]):
                cat_ious[cat] = np.mean(self._shape_ious[cat])
        mean_class_ious = np.mean(list(cat_ious.values()))
        return cat_ious, mean_class_ious, np.mean(instance_ious)

    def get_metrics(self, verbose=False) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)
        metrics["{}_Cmiou".format(self._stage)] = self._Cmiou * 100
        metrics["{}_Imiou".format(self._stage)] = self._Imiou * 100
        if verbose:
            metrics["{}_Imiou_per_class".format(self._stage)] = self._miou_per_class
        return metrics

    @property
    def metric_func(self):
        self._metric_func = {"Cmiou": max, "Imiou": max, "loss": min}
        return self._metric_func
