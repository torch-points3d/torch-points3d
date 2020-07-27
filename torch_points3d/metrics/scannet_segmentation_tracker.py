import os.path as osp
from typing import Dict, Any
import logging
import numpy as np
import torch
from torch_geometric.nn.unpool import knn_interpolate

from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.models import model_interface

log = logging.getLogger(__name__)


class ScannetSegmentationTracker(SegmentationTracker):
    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._full_confusion_matrix = ConfusionMatrix(self._num_classes)
        self._raw_datas = {}
        self._votes = {}
        self._vote_counts = {}
        self._full_preds = {}
        self._full_acc = None

    def track(self, model: model_interface.TrackerInterface, full_res=False, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)

        # Set conv type
        self._conv_type = model.conv_type

        # Train mode or low res, nothing special to do
        if not full_res or self._stage == "train" or kwargs.get("data") is None:
            return

        self._vote(kwargs.get("data"), model.get_output())

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)
        if self._full_acc:
            metrics["{}_full_acc".format(self._stage)] = self._full_acc
            metrics["{}_full_macc".format(self._stage)] = self._full_macc
            metrics["{}_full_miou".format(self._stage)] = self._full_miou
        return metrics

    def finalise(self, full_res=False, make_submission=False, **kwargs):
        if not full_res and not make_submission:
            return

        self._predict_full_res()

        # Compute full res metrics
        if self._dataset.has_labels(self._stage):
            for scan_id in self._full_preds:
                full_labels = self._raw_datas[scan_id].y
                # Mask ignored labels
                mask = full_labels != self._ignore_label
                full_labels = full_labels[mask]
                full_preds = self._full_preds[scan_id].cpu()[mask].numpy()
                self._full_confusion_matrix.count_predicted_batch(full_labels, full_preds)

            self._full_acc = 100 * self._full_confusion_matrix.get_overall_accuracy()
            self._full_macc = 100 * self._full_confusion_matrix.get_mean_class_accuracy()
            self._full_miou = 100 * self._full_confusion_matrix.get_average_intersection_union()

        # Save files to disk
        if make_submission and self._stage == "test":
            self._make_submission()

    def _make_submission(self):
        orginal_class_ids = np.asarray(self._dataset.train_dataset.valid_class_idx)
        path_to_submission = self._dataset.path_to_submission
        for scan_id in self._full_preds:
            full_pred = self._full_preds[scan_id].cpu().numpy().astype(np.int8)
            full_pred = orginal_class_ids[full_pred]  # remap labels to original labels between 0 and 40
            scan_name = self._raw_datas[scan_id].scan_name
            path_file = osp.join(path_to_submission, "{}.txt".format(scan_name))
            np.savetxt(path_file, full_pred, delimiter="/n", fmt="%d")

    def _vote(self, data, output):
        """ Populates scores for the points in data

        Parameters
        ----------
        data : Data
            should contain `pos` and `SaveOriginalPosId.KEY` keys
        output : torch.Tensor
            probablities out of the model, shape: [N,nb_classes]
        """
        id_scans = data.id_scan.squeeze()
        if self._conv_type == "DENSE":
            batch_size = len(id_scans)
            output = output.view(batch_size, -1, output.shape[-1])

        for idx_batch, id_scan in enumerate(id_scans):
            # First time we see this scan
            if id_scan not in self._raw_datas:
                raw_data = self._dataset.get_raw_data(self._stage, id_scan, remap_labels=True)
                self._raw_datas[id_scan] = raw_data
                self._vote_counts[id_scan] = torch.zeros(raw_data.pos.shape[0], dtype=torch.int)
                self._votes[id_scan] = torch.zeros((raw_data.pos.shape[0], self._num_classes), dtype=torch.float)
            else:
                raw_data = self._raw_datas[id_scan]

            batch_mask = idx_batch
            if self._conv_type != "DENSE":
                batch_mask = data.batch == idx_batch
            idx = data[SaveOriginalPosId.KEY][batch_mask]

            self._votes[id_scan][idx] += output[batch_mask].cpu()
            self._vote_counts[id_scan][idx] += 1

    def _predict_full_res(self):
        """ Predict full resolution results based on votes """
        for id_scan in self._votes:
            has_prediction = self._vote_counts[id_scan] > 0
            self._votes[id_scan][has_prediction] /= self._vote_counts[id_scan][has_prediction].unsqueeze(-1)

            # Upsample and predict
            full_pred = knn_interpolate(
                self._votes[id_scan][has_prediction],
                self._raw_datas[id_scan].pos[has_prediction],
                self._raw_datas[id_scan].pos,
                k=1,
            )
            self._full_preds[id_scan] = full_pred.argmax(-1)
