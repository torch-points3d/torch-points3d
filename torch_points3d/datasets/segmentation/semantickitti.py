import os
from glob import glob
import numpy as np
import multiprocessing
import logging
import torch

from torch_geometric.data import Dataset, Data

from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.segmentation.kitti_config import *
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker

log = logging.getLogger(__name__)


class SemanticKitti(Dataset):
    r"""SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences"
    from the <https://arxiv.org/pdf/1904.01416.pdf> paper, 
    containing about 21 lidar scan sequences with dense point-wise annotations.
    
    root dir should be structured as
    rootdir
        └── sequences/
            ├── 00/
            │   ├── labels/
            │   │     ├ 000000.label
            │   │     └ 000001.label
            │   └── velodyne/
            │         ├ 000000.bin
            │         └ 000001.bin
            ├── 01/
            ├── 02/
            .
            .
            .
            └── 21/
    Args:
        root (string): Root directory of the dataset.
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"trainval"`, loads the training and validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"trainval"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """

    LABELS = LABELS
    COLOR_MAP = COLOR_MAP
    CONTENT = CONTENT
    REMAPPING_MAP = REMAPPING_MAP
    LEARNING_MAP_INV = LEARNING_MAP_INV
    SPLIT = SPLIT
    AVAILABLE_SPLITS = ["train", "val", "test", "trainval"]

    def __init__(self, root, split="trainval", transform=None, process_workers=1, pre_transform=None):
        assert self.REMAPPING_MAP[0] == IGNORE_LABEL  # Make sure we have the same convention for unlabelled data
        self.use_multiprocessing = process_workers > 1
        self.process_workers = process_workers

        super().__init__(root, transform=transform, pre_transform=pre_transform)
        if split == "train":
            self._scans = glob(os.path.join(self.processed_paths[0], "*.pt"))
        elif split == "val":
            self._scans = glob(os.path.join(self.processed_paths[1], "*.pt"))
        elif split == "test":
            self._scans = glob(os.path.join(self.processed_paths[2], "*.pt"))
        elif split == "trainval":
            self._scans = glob(os.path.join(self.processed_paths[0], "*.pt")) + glob(
                os.path.join(self.processed_paths[1], "*.pt")
            )
        else:
            raise ValueError("Split %s not recognised" % split)

    @property
    def raw_file_names(self):
        return ["sequences"]

    @property
    def processed_file_names(self):
        return [s for s in self.AVAILABLE_SPLITS[:-1]]

    def _load_paths(self, seqs):
        scan_paths = []
        label_path = []
        for seq in seqs:
            scan_paths.extend(
                sorted(glob(os.path.join(self.raw_paths[0], "{0:02d}".format(int(seq)), "velodyne", "*.bin")))
            )
            label_path.extend(
                sorted(glob(os.path.join(self.raw_paths[0], "{0:02d}".format(int(seq)), "labels", "*.label")))
            )

        if len(label_path) == 0:
            label_path = [None for i in range(len(scan_paths))]
        if len(label_path) > 0 and len(scan_paths) != len(label_path):
            raise ValueError((f"number of scans {len(scan_paths)} not equal to number of labels {len(label_path)}"))

        return scan_paths, label_path

    @staticmethod
    def read_raw(scan_file, label_file=None):
        scan = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)
        data = Data(pos=torch.tensor(scan[:, :3]), x=torch.tensor(scan[:, 3]).reshape(-1, 1),)
        if label_file:
            label = np.fromfile(label_file, dtype=np.uint32).astype(np.int32)
            assert scan.shape[0] == label.shape[0]
            semantic_label = label & 0xFFFF
            instance_label = label >> 16
            data.y = torch.tensor(semantic_label).long()
            data.instance_labels = torch.tensor(instance_label).long()
        return data

    @staticmethod
    def process_one(scan_file, label_file, transform, out_file):
        data = SemanticKitti.read_raw(scan_file, label_file)
        if transform:
            data = transform(data)
        log.info("Processed file %s, nb points = %i", os.path.basename(out_file), data.pos.shape[0])
        torch.save(data, out_file)

    def get(self, idx):
        data = torch.load(self._scans[idx])
        if data.y is not None:
            data.y = self._remap_labels(data.y)
        return data

    def process(self):
        for i, split in enumerate(self.AVAILABLE_SPLITS[:-1]):
            if os.path.exists(self.processed_paths[i]):
                continue
            os.makedirs(self.processed_paths[i])

            seqs = self.SPLIT[split]
            scan_paths, label_paths = self._load_paths(seqs)
            scan_names = []
            for scan in scan_paths:
                scan = os.path.splitext(scan)[0]
                seq, _, scan_id = scan.split(os.path.sep)[-3:]
                scan_names.append("{}_{}".format(seq, scan_id))

            out_files = [os.path.join(self.processed_paths[i], "{}.pt".format(scan_name)) for scan_name in scan_names]
            args = zip(scan_paths, label_paths, [self.pre_transform for i in range(len(scan_paths))], out_files)
            if self.use_multiprocessing:
                with multiprocessing.Pool(processes=self.process_workers) as pool:
                    pool.starmap(self.process_one, args)
            else:
                for arg in args:
                    self.process_one(*arg)

    def len(self):
        return len(self._scans)

    def download(self):
        if len(os.listdir(self.raw_dir)) == 0:
            url = "http://semantic-kitti.org/"
            print(f"please download the dataset from {url} with the following folder structure")
            print(
                """
                    rootdir
                        └── sequences/
                            ├── 00/
                            │   ├── labels/
                            │   │     ├ 000000.label
                            │   │     └ 000001.label
                            │   └── velodyne/
                            │         ├ 000000.bin
                            │         └ 000001.bin
                            ├── 01/
                            ├── 02/
                            .
                            .
                            .
                          
                            └── 21/
                """
            )

    def _remap_labels(self, semantic_label):
        """ Remaps labels to [0 ; num_labels -1]. Can be overriden.
        """
        new_labels = semantic_label.clone()
        for source, target in self.REMAPPING_MAP.items():
            mask = semantic_label == source
            new_labels[mask] = target

        return new_labels

    @property
    def num_classes(self):
        return 19


class SemanticKittiDataset(BaseDataset):
    """ Wrapper around Semantic Kitti that creates train and test datasets.
    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain
            - root,
            - split,
            - transform,
            - pre_transform
            - process_workers
    """

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        process_workers: int = dataset_opt.process_workers if dataset_opt.process_workers else 0
        self.train_dataset = SemanticKitti(
            self._data_path,
            split="train",
            transform=self.train_transform,
            pre_transform=self.pre_transform,
            process_workers=process_workers,
        )

        self.val_dataset = SemanticKitti(
            self._data_path,
            split="val",
            transform=self.val_transform,
            pre_transform=self.pre_transform,
            process_workers=process_workers,
        )

        self.test_dataset = SemanticKitti(
            self._data_path,
            split="test",
            transform=self.test_transform,
            pre_transform=self.pre_transform,
            process_workers=process_workers,
        )

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker
        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)


if __name__ == "__main__":
    DIR = os.path.dirname(os.path.realpath(__file__))
    dataroot = os.path.join(DIR, "..", "..", "data", "kitti")
    SemanticKitti(
        dataroot, split="train", process_workers=10,
    )

