import os
import os.path as osp
import logging
import torch

from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.models import model_interface

log = logging.getLogger(__name__)


class ScannetSegmentationTracker(SegmentationTracker):
    def make_submission(self, data, outputs, dataset, conv_type):
        root = dataset.root
        scan_tests_mapping = dataset.scan_tests_mapping
        path_to_submission = osp.join(root, "unzip_root")
        if os.path.exists(path_to_submission):
            os.makedirs(path_to_submission)

        id_scans = data.id_scan.squeeze()

        if conv_type == "DENSE":
            predictions = torch.stack(outputs).mean(0).argmax(dim=-1).view((data.pos.shape[0], -1))

            for idx_batch, id_scan in enumerate(id_scans):
                predictions[idx_batch]
                scannet_dir = osp.join(dataset.raw_dir, "scans_test")
                scan_name = scan_tests_mapping[id_scan.item()]
                dataset.read_one_test_scan(scannet_dir, scan_name, 0, dataset.normalize_rgb)
                import pdb

                pdb.set_trace()
