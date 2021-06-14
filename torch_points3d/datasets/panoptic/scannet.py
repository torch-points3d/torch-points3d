import os
import numpy as np
import torch
import logging
from torch_geometric.data import InMemoryDataset
from torch_points3d.datasets.segmentation.scannet import Scannet
from torch_points3d.metrics.panoptic_tracker import PanopticTracker
from torch_points3d.datasets.base_dataset import BaseDataset, save_used_properties
from torch_points3d.datasets.panoptic.utils import set_extra_labels

DIR = os.path.dirname(os.path.realpath(__file__))

log = logging.getLogger(__name__)


class ScannetPanoptic(Scannet):
    NYU40IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
    NUM_MAX_OBJECTS = 64

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.STUFFCLASSES = torch.tensor([i for i in self.VALID_CLASS_IDS if i not in self.NYU40IDS])
        self.NYU40ID2CLASS = {nyu40id: i for i, nyu40id in enumerate(list(self.NYU40IDS))}

    def __getitem__(self, idx):
        """
        Data object contains:
            pos - points
            x - features
        """
        if not isinstance(idx, int):
            raise ValueError("Only integer indices supported")

        # Get raw data and apply transforms
        data = super().__getitem__(idx)

        # Extract instance and box labels
        self._set_extra_labels(data)
        data.y = super()._remap_labels(data.y)
        return data

    def _set_extra_labels(self, data):
        return set_extra_labels(data, self.NYU40ID2CLASS, self.NUM_MAX_OBJECTS)

    def _remap_labels(self, semantic_label):
        return semantic_label

    @property
    def stuff_classes(self):
        return super()._remap_labels(self.STUFFCLASSES)

    def process(self):
        if self.is_test:
            pass
        super().process()

    def download(self):
        if self.is_test:
            pass
        super().download()


class ScannetDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        # Update to OmegaConf 2.0
        use_instance_labels: bool = dataset_opt.get('use_instance_labels')
        donotcare_class_ids: [] = list(dataset_opt.get('donotcare_class_ids', []))
        max_num_point: int = dataset_opt.get('max_num_point', None)
        is_test: bool = dataset_opt.get('is_test', False)

        self.train_dataset = ScannetPanoptic(
            self._data_path,
            split="train",
            pre_transform=self.pre_transform,
            transform=self.train_transform,
            version=dataset_opt.version,
            use_instance_labels=use_instance_labels,
            use_instance_bboxes=False,
            donotcare_class_ids=donotcare_class_ids,
            max_num_point=max_num_point,
            is_test=is_test,
        )

        self.val_dataset = ScannetPanoptic(
            self._data_path,
            split="val",
            transform=self.val_transform,
            pre_transform=self.pre_transform,
            version=dataset_opt.version,
            use_instance_labels=use_instance_labels,
            use_instance_bboxes=False,
            donotcare_class_ids=donotcare_class_ids,
            max_num_point=max_num_point,
            is_test=is_test,
        )

    @property  # type: ignore
    @save_used_properties
    def stuff_classes(self):
        """ Returns a list of classes that are not instances
        """
        return self.train_dataset.stuff_classes

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
        Returns:
            [BaseTracker] -- tracker
        """
        return PanopticTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
