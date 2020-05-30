import os
import os.path as osp
import logging
import numpy as np
import torch

from torch_geometric.nn.unpool import knn_interpolate
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.models import model_interface

log = logging.getLogger(__name__)


class ScannetSegmentationTracker(SegmentationTracker):
    def make_submission(self, datas, outputs, dataset, conv_type):
        root = dataset.root
        scan_tests_mapping = dataset.scan_tests_mapping
        path_to_submission = osp.join(root, "unzip_root")
        if not os.path.exists(path_to_submission):
            os.makedirs(path_to_submission)

        id_scans = datas[0].id_scan.squeeze()
        num_classes = outputs[0].shape[-1]
        num_votes = len(outputs)

        if conv_type == "DENSE":

            predictions = torch.stack(outputs).view((datas[0].pos.shape[0], num_votes, -1, num_classes))

            for idx_batch, id_scan in enumerate(id_scans):
                scannet_dir = osp.join(dataset.raw_dir, "scans_test")
                scan_name = scan_tests_mapping[id_scan.item()]
                log.info("PREDICTION FOR FILE: {}".format(scan_name))
                raw_data = dataset.read_one_test_scan(scannet_dir, scan_name, 0, dataset.normalize_rgb)

                votes_counts = torch.zeros(raw_data.pos.shape[0], dtype=torch.int)
                votes = torch.zeros((raw_data.pos.shape[0], num_classes), dtype=torch.float)
                predictions[idx_batch]

                for idx_votes in range(num_votes):
                    idx = datas[idx_votes][SaveOriginalPosId.KEY][idx_batch]
                    votes[idx] += predictions[idx_batch][idx_votes]
                    votes_counts[idx] += 1

                has_prediction = votes_counts > 0
                votes[has_prediction] /= votes_counts[has_prediction].unsqueeze(-1)

                full_pred = knn_interpolate(votes[has_prediction], raw_data.pos[has_prediction], raw_data.pos, k=1,)

                full_pred = full_pred.argmax(-1).cpu().numpy().astype(np.int8)

                path_file = osp.join(path_to_submission, "{}.txt".format(scan_name))

                np.savetxt(path_file, full_pred, delimiter="/n", fmt="%d")
