import os.path as osp
from typing import Dict
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
        self._full_res = False
        self._full_confusion_matrix = ConfusionMatrix(self._num_classes)

    def track(self, model: model_interface.TrackerInterface, full_res=False, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)

        # Train mode or low res, nothing special to do
        if not full_res or self._stage == "train" or kwargs.get("data") is None:
            return

        self._full_res = full_res
        self._conv_type = model.conv_type
        _, full_preds, full_labels = self._outputs_to_full_res([kwargs.get("data")], [model.get_output()])

        assert [fp.shape for fp in full_preds] == [fl.shape for fl in full_labels]

        for scan_id in full_preds:
            self._full_confusion_matrix.count_predicted_batch(full_labels[scan_id], full_preds[scan_id].cpu().numpy())

        self._full_acc = 100 * self._full_confusion_matrix.get_overall_accuracy()
        self._full_macc = 100 * self._full_confusion_matrix.get_mean_class_accuracy()
        self._full_miou = 100 * self._full_confusion_matrix.get_average_intersection_union()

    def get_metrics(self, verbose=False) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)
        if self._full_res:
            metrics["{}_full_acc".format(self._stage)] = self._full_acc
            metrics["{}_full_macc".format(self._stage)] = self._full_macc
            metrics["{}_full_miou".format(self._stage)] = self._full_miou
        return metrics

    def make_submission(self, datas, outputs, conv_type):
        self._conv_type = conv_type
        orginal_class_ids = np.asarray(self._dataset.train_dataset.valid_class_idx)
        path_to_submission = self._dataset.path_to_submission
        scan_names, full_preds, _ = self._outputs_to_full_res(datas, outputs)
        for scan_id in scan_names:
            full_pred = full_preds[scan_id].cpu().numpy().astype(np.int8)
            full_pred = orginal_class_ids[full_pred]  # remap labels to original labels between 0 and 40
            path_file = osp.join(path_to_submission, "{}.txt".format(scan_names[scan_id]))
            np.savetxt(path_file, full_pred, delimiter="/n", fmt="%d")

    def _outputs_to_full_res(self, datas, outputs):
        raw_datas = {}
        votes = {}
        vote_counts = {}
        full_labels = {}
        full_preds = {}

        # Gather votes
        for i, data in enumerate(datas):
            id_scans = data.id_scan.squeeze()
            output = outputs[i]
            if self._conv_type == "DENSE":
                batch_size = len(id_scans)
                output = output.view(batch_size, -1, output.shape[-1])

            for idx_batch, id_scan in enumerate(id_scans):
                # First time we see this scan
                if id_scan not in raw_datas:
                    raw_data = self._dataset.get_raw_data(self._stage, id_scan)
                    raw_datas[id_scan] = raw_data
                    vote_counts[id_scan] = torch.zeros(raw_data.pos.shape[0], dtype=torch.int)
                    votes[id_scan] = torch.zeros((raw_data.pos.shape[0], self._num_classes), dtype=torch.float)
                    if self._dataset.dataset_has_labels(self._stage):
                        full_labels[id_scan] = raw_data.y
                else:
                    raw_data = raw_datas[id_scan]

                batch_mask = idx_batch
                if self._conv_type != "DENSE":
                    batch_mask = data.batch == idx_batch
                idx = data[SaveOriginalPosId.KEY][batch_mask]

                votes[id_scan][idx] += output[batch_mask]
                vote_counts[id_scan][idx] += 1

        # Predict and upsample
        scan_names = {}
        for id_scan in votes:
            has_prediction = vote_counts[id_scan] > 0
            votes[id_scan][has_prediction] /= vote_counts[id_scan][has_prediction].unsqueeze(-1)

            full_pred = knn_interpolate(
                votes[id_scan][has_prediction], raw_datas[id_scan].pos[has_prediction], raw_datas[id_scan].pos, k=1
            )
            full_preds[id_scan] = full_pred.argmax(-1)
            scan_names[id_scan] = raw_datas[id_scan].scan_name
        return scan_names, full_preds, full_labels
