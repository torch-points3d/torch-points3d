import os
from glob import glob
import numpy as np

from torch_geometric.data import Dataset, Data
from torch_points3d.datasets.segmentation.kitti_config import *
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker

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
    LEARNING_MAP = LEARNING_MAP
    LEARNING_MAP_INV = LEARNING_MAP_INV
    LEARNING_IGNORE = LEARNING_IGNORE
    SPLIT = SPLIT
    AVAILABLE_SPLITS = ['train', 'val', 'test', 'trainval']

    def __init__(
            self,
            root,
            split="trainval",
            transform=None,
            is_test=False,
            normalise=True
        ):
        super(SemanticKittiDataset, self).__init__(
            root, transform)
        
        self._normalise = normalise
        self._populate_colormap()

        if split in AVAILABLE_SPLITS:
            self._load_paths(split)
        else:
            raise ValueError(
                (f"Split {split} found, but expected either " "train, val, trainval or test"))

    def _load_paths(self, split):
        self._scan_paths = []
        self._label_path = []
        if split == "trainval":
            seqs = self.SPLIT["train"] + self.SPLIT["val"]
        else:
            seqs = self.SPLIT[split]
        for seq in seqs:
            self._scan_paths.extend(sorted(glob(os.path.join(self.root, "{0:02d}".format(int(seq)), "velodyne", '*.bin'))))
            self._label_path.extend(sorted(glob(os.path.join(self.root, "{0:02d}".format(int(seq)), "labels", '*.label'))))
        if len(self._scan_paths) != len(self._label_path):
            raise ValueError((f"number of scans {len(self._scan_paths)} not equal to number of labels {len(self._label_path)}"))        

    @staticmethod
    def open_scan(file_name):
        """
        open scan file - [x, y, z, remissions]
        returns point coordinates, remissions
        """
        scan = np.fromfile(file_name, dtype=np.float32)
        scan = scan.reshape((-1, 4)) # just for the sake of it
        return scan
    
    @staticmethod
    def open_label(file_name):
        """
        open label file - [semantic label(first 16 bits), instance label(last 16 bits)]
        returns semantic label, instance label
        """
        label = np.fromfile(file_name, dtype=np.uint32)
        label = label.reshape((-1)) # again for the sake of it
        semantic_label = label & 0xFFFF
        instance_label = label >> 16
        return semantic_label, instance_label

    def __getitem__(self, idx):
        scan = SemanticKitti.open_scan(self._scan_paths[idx])
        label = None
        if self._label_path is not None:
            label, _ = SemanticKitti.open_label(self._label_path[idx])
        data = Data(scan=scan, label=label)
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def download(self):
        if len(os.listdir(self.raw_dir)) == 0:
            url = "http://semantic-kitti.org/"
            print(f"please download the dataset from {url} with the following folder structure")
            print("""
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
                """)
    
    def convert_labels(self, labels, reduced=True):
        if reduced:
            return np.vectorize(self.LEARNING_MAP.get)(labels)
        else:
            return np.vectorize(self.LEARNING_MAP_INV.get)(labels)
    
    def populate_colormap(self):
        self.CMAP = np.zeros((max(list(self.LEARNING_MAP.keys()))+1, 3), dtype=np.float32)
        for key, value in self.COLOR_MAP.items():
            value = [value[i] for i in [2,1,0]]
            self.CMAP[key] = np.array(value, np.float32) / 255.0
    
    def colorise(self, labels):
        if len(labels.shape)>1:
            proj_color_labels = np.zeros((self.H, self.W, 3), dtype=np.float)
            proj_labels = self.convert_labels(labels, False) # sanity check
            proj_color_labels = self.CMAP[labels]
            return proj_color_labels
        else:
            labels = self.convert_labels(labels, False) # sanity check
            color_labels = self.CMAP[labels]
            return color_labels

class SemanticKittiDataset(BaseDataset):
    """ Wrapper around Semantic Kitti that creates train and test datasets.

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - root,
            - split,
            - transform,
            - is_test,
            - normalise
    """

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        try:
            is_test = dataset_opt.get("is_test", False)
            normalise = dataset_opt.get("normalise", True)

        self.train_dataset = SemanticKitti(
            self._data_path,
            split="train",
            transform=self.train_transform,
            is_test=is_test,
            normalise=normalise
        )

        self.val_dataset = SemanticKitti(
            self._data_path,
            split="val",
            transform=self.val_transform,
            is_test=is_test,
            normalise=normalise
        )

        self.test_dataset = SemanticKitti(
            self._data_path,
            split="test",
            transform=self.test_transform,
            is_test=True,
            normalise=normalise
        )

        self._categories = list(self.train_dataset.LABELS.values)
    
    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)

