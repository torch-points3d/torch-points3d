import os.path as osp
from typing import Dict
import logging
import numpy as np
import torch
from typing import List
from torch_geometric.data import Data
from torch_geometric.nn.unpool import knn_interpolate
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.models import model_interface

log = logging.getLogger(__name__)


class ScannetSegmentationTracker(SegmentationTracker):
    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._full_res = False

    def track(self, model: model_interface.TrackerInterface, full_res=True, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        # Train mode or low res, nothing special to do
        if not full_res:
            super().track(model)
            return

        if self._stage == "train":
            super().track(model)
            return

        if kwargs.get("data") is None:
            super().track(model)
            return

        self._full_res = full_res
        self._conv_type = model.conv_type

        datas: List[Data] = [kwargs.get("data")]
        datas[0].id_scan.squeeze()

        _, full_preds, full_labels = self.outputs_to_full_res([kwargs.get("data")], [model.get_output()])

        assert [fp.shape for fp in full_preds] == [fl.shape for fl in full_labels]

        for full_label, full_pred in zip(full_labels, full_preds):
            self._confusion_matrix.count_predicted_batch(full_label, full_pred.cpu().numpy())

        self._acc = 100 * self._confusion_matrix.get_overall_accuracy()
        self._macc = 100 * self._confusion_matrix.get_mean_class_accuracy()
        self._miou = 100 * self._confusion_matrix.get_average_intersection_union()

    def get_metrics(self, verbose=False) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        if self._full_res:
            self._stage += "_full"
        return super().get_metrics(verbose)

    def make_submission(self, datas, outputs):
        path_to_submission = self._dataset.path_to_submission

        scan_names, full_preds, _ = self.outputs_to_full_res(datas, outputs)

        for scan_name, full_pred in zip(scan_names, full_preds):
            full_pred = full_pred.cpu().numpy().astype(np.int8)
            path_file = osp.join(path_to_submission, "{}.txt".format(scan_name))
            np.savetxt(path_file, full_pred, delimiter="/n", fmt="%d")

    def outputs_to_full_res(self, datas, outputs):

        id_scans = datas[0].id_scan.squeeze()
        num_votes = len(outputs)
        scan_names = []
        full_preds = []
        full_labels = []

        predictions = torch.stack(outputs)

        if self._conv_type == "DENSE":
            predictions = predictions.view((datas[0].pos.shape[0], num_votes, -1, self._num_classes))

        for idx_batch, id_scan in enumerate(id_scans):
            raw_data = self._dataset.get_raw_data(self._stage, id_scan)
            scan_names.append(raw_data.scan_name)

            if self._dataset.dataset_has_labels(self._stage):
                full_labels.append(raw_data.y)

            votes_counts = torch.zeros(raw_data.pos.shape[0], dtype=torch.int)
            votes = torch.zeros((raw_data.pos.shape[0], self._num_classes), dtype=torch.float)

            batch_mask = idx_batch
            for idx_votes in range(num_votes):
                if self._conv_type != "DENSE":
                    batch_mask = datas[idx_votes].batch == idx_batch
                idx = datas[idx_votes][SaveOriginalPosId.KEY][batch_mask]

                if self._conv_type == "DENSE":
                    votes[idx] += predictions[idx_batch][idx_votes]
                else:
                    votes[idx] += predictions[idx_votes][batch_mask]
                votes_counts[idx] += 1

            has_prediction = votes_counts > 0
            votes[has_prediction] /= votes_counts[has_prediction].unsqueeze(-1)

            full_pred = knn_interpolate(votes[has_prediction], raw_data.pos[has_prediction], raw_data.pos, k=1)
            full_preds.append(full_pred.argmax(-1))

        return scan_names, full_preds, full_labels
