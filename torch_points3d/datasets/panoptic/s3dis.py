import numpy as np
import torch
import random

from torch_points3d.datasets.base_dataset import BaseDataset, save_used_properties
from torch_points3d.datasets.segmentation.s3dis import S3DISSphere, S3DISCylinder, INV_OBJECT_LABEL
import torch_points3d.core.data_transform as cT
from torch_points3d.metrics.panoptic_tracker import PanopticTracker
from torch_points3d.datasets.panoptic.utils import set_extra_labels

CLASSES_INV = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "chair",
    8: "table",
    9: "bookcase",
    10: "sofa",
    11: "board",
    12: "clutter",
}


class PanopticS3DISBase:
    INSTANCE_CLASSES = CLASSES_INV.keys()
    NUM_MAX_OBJECTS = 64

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
        return data

    def _set_extra_labels(self, data):
        return set_extra_labels(data, self.INSTANCE_CLASSES, self.NUM_MAX_OBJECTS)

    @property
    def stuff_classes(self):
        return torch.tensor([])


class PanopticS3DISSphere(PanopticS3DISBase, S3DISSphere):
    def process(self):
        super().process()

    def download(self):
        super().download()


class PanopticS3DISCylinder(PanopticS3DISBase, S3DISCylinder):
    def process(self):
        super().process()

    def download(self):
        super().download()


class S3DISFusedDataset(BaseDataset):
    """ Wrapper around S3DISSphere that creates train and test datasets.

    http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - fold: test_area parameter
            - pre_collate_transform
            - train_transforms
            - test_transforms
    """

    INV_OBJECT_LABEL = INV_OBJECT_LABEL

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        sampling_format = dataset_opt.get("sampling_format", "cylinder")
        dataset_cls = PanopticS3DISCylinder if sampling_format == "cylinder" else PanopticS3DISSphere

        self.train_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=3000,
            test_area=self.dataset_opt.fold,
            split="train",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.train_transform,
            keep_instance=True,
        )

        self.val_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="val",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.val_transform,
            keep_instance=True,
        )
        self.test_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="test",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.test_transform,
            keep_instance=True,
        )

        if dataset_opt.class_weight_method:
            self.add_weights(class_weight_method=dataset_opt.class_weight_method)

    @property
    def test_data(self):
        return self.test_dataset[0].raw_test_data

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
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """

        return PanopticTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)

