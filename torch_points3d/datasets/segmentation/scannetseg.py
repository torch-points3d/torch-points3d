import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
import os.path as osp
import os
from plyfile import PlyData, PlyElement
import pandas as pd
from pathlib import Path
import multiprocessing
import logging

from torch_points3d.datasets.base_dataset import BaseDataset
import torch_points3d.core.data_transform as cT
from . import IGNORE_LABEL

CLASS_LABELS = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)
VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

SCANNET_COLOR_MAP = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}

log = logging.getLogger(__name__)


def read_plyfile(filepath):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(filepath, "rb") as f:
        plydata = PlyData.read(f)
    if plydata.elements:
        return pd.DataFrame(plydata.elements[0].data).values


class Scannet(InMemoryDataset):
    VALID_CLASS_IDS = VALID_CLASS_IDS
    SCANNET_COLOR_MAP = SCANNET_COLOR_MAP
    CLASS_LABELS = CLASS_LABELS
    SPLITS = ["train", "val", "test"]
    NUM_CLASSES = 41

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        process_workers=4,
        types=[".txt", "_vh_clean_2.ply", "_vh_clean_2.0.010000.segs.json", ".aggregation.json"],
        normalize_rgb=True,
    ):
        self.use_multiprocessing = process_workers > 1
        self.process_workers = process_workers
        super().__init__(root, transform, pre_transform, pre_filter)
        if split == "train":
            path = self.processed_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
        else:
            raise ValueError((f"Split {split} found, but expected either " "train, val, or test"))

        self.data, self.slices = torch.load(path)

        if split != "test":
            self.data.y = self._remap_labels(self.data.y)
            self.has_labels = True
        else:
            self.has_labels = False

    @property
    def raw_file_names(self):
        return ["metadata", "scans", "scannetv2-labels.combined.tsv"]

    @property
    def processed_file_names(self):
        return ["{}.pt".format(s,) for s in Scannet.SPLITS]

    def read_from_metadata(self):
        metadata_path = osp.join(self.raw_dir, "metadata")
        split_files = ["scannetv2_{}.txt".format(s) for s in Scannet.SPLITS]
        self.scan_names = []
        for sf in split_files:
            f = open(osp.join(metadata_path, sf))
            self.scan_names.append(sorted([line.rstrip() for line in f]))
            f.close()

        for idx_split, split in enumerate(Scannet.SPLITS):
            idx_mapping = {idx: scan_name for idx, scan_name in enumerate(self.scan_names[idx_split])}
            setattr(self, "MAPPING_IDX_TO_SCAN_{}_NAMES".format(split.upper()), idx_mapping)

    @staticmethod
    def read_one_scan(scannet_dir, scan_name):
        log.info(scan_name)
        mesh_file = Path(osp.join(scannet_dir, scan_name, scan_name + "_vh_clean_2.ply"))
        label_file = Path(osp.join(scannet_dir, scan_name, scan_name + "_vh_clean_2.labels.ply"))
        pointcloud = read_plyfile(mesh_file)

        if label_file.is_file():
            label = read_plyfile(label_file)
            # Sanity check that the pointcloud and its label has same vertices.
            assert pointcloud.shape[0] == label.shape[0]
            assert np.allclose(pointcloud[:, :3], label[:, :3])
        else:  # Label may not exist in test case.
            label = None

        data = Data(pos=torch.tensor(pointcloud[:, :3]), rgb=torch.tensor(pointcloud[:, 3:6]))
        data.rgb = data.rgb.float() / 255.0
        if label is not None:
            data.y = torch.tensor(label[:, -1]).long()
        log.info("{} - {}".format(scan_name, data))
        return data

    def process(self):
        self.read_from_metadata()
        for idx_split, split in enumerate(self.SPLITS):
            if os.path.exists(self.processed_paths[idx_split]):
                continue
            scannet_dir = osp.join(self.raw_dir, "scans" if split in ["train", "val"] else "scans_test")
            args = [(scannet_dir, scan_name) for scan_name in self.scan_names[idx_split]]
            if self.use_multiprocessing:
                with multiprocessing.Pool(processes=self.process_workers) as pool:
                    datas = pool.starmap(Scannet.read_one_scan, args)
            else:
                datas = []
                for arg in args:
                    data = Scannet.read_one_scan(*arg)
                    datas.append(data)

            if self.pre_transform:
                datas = [self.pre_transform(data) for data in datas]

            log.info("SAVING TO {}".format(self.processed_paths[idx_split]))
            torch.save(self.collate(datas), self.processed_paths[idx_split])

    def _remap_labels(self, semantic_label):
        """ Remaps labels to [0 ; num_labels -1]. Can be overriden.
        """
        new_labels = semantic_label.clone()
        mapping_dict = {indice: idx for idx, indice in enumerate(self.VALID_CLASS_IDS)}
        for idx in range(self.NUM_CLASSES):
            if idx not in mapping_dict:
                mapping_dict[idx] = IGNORE_LABEL
        for source, target in mapping_dict.items():
            mask = semantic_label == source
            new_labels[mask] = target

        broken_labels = new_labels >= len(self.VALID_CLASS_IDS)
        new_labels[broken_labels] = IGNORE_LABEL

        return new_labels

    @property
    def path_to_submission(self):
        root = os.getcwd()
        path_to_submission = os.path.join(root, "submission_labels")
        if not os.path.exists(path_to_submission):
            os.makedirs(path_to_submission)
        return path_to_submission

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, len(self))


class ScannetSegDataset(BaseDataset):
    """ Wrapper around Scannet that creates train and test datasets.

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - version
            - max_num_point (optional)
            - use_instance_labels (optional)
            - use_instance_bboxes (optional)
            - donotcare_class_ids (optional)
            - pre_transforms (optional)
            - train_transforms (optional)
            - val_transforms (optional)
    """

    SPLITS = Scannet.SPLITS

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        process_workers: int = dataset_opt.process_workers if dataset_opt.process_workers else 0

        self.train_dataset = Scannet(
            self._data_path,
            split="train",
            pre_transform=self.pre_transform,
            transform=self.train_transform,
            process_workers=process_workers,
        )

        self.val_dataset = Scannet(
            self._data_path,
            split="val",
            transform=self.val_transform,
            pre_transform=self.pre_transform,
            process_workers=process_workers,
        )

        self.test_dataset = Scannet(
            self._data_path,
            split="test",
            transform=self.val_transform,
            pre_transform=self.pre_transform,
            process_workers=process_workers,
        )

    @property
    def path_to_submission(self):
        return self.train_dataset.path_to_submission

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            dataset {[type]}
            wandb_log - Log using weight and biases
        Returns:
            [BaseTracker] -- tracker
        """
        from torch_points3d.metrics.scannet_segmentation_tracker import ScannetSegmentationTracker

        return ScannetSegmentationTracker(
            self, wandb_log=wandb_log, use_tensorboard=tensorboard_log, ignore_label=IGNORE_LABEL
        )
