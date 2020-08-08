import numpy as np
import torch
import random

from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.segmentation.s3dis import (
    S3DISSphere as SegmentationS3DISSphere, 
    add_weights, 
    INV_OBJECT_LABEL
)
import torch_points3d.core.data_transform as cT
from sklearn.neighbors import NearestNeighbors, KDTree
from torch_points3d.metrics.panoptic_tracker import PanopticTracker
from torch_points3d.core.data_transform import CylinderSampling, GridCylinderSampling

STUFF_CLASSES_INV = {
    # 0: "ceiling",
    # 1: "floor",
    # 2: "wall",
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

class PanopticS3DISSphere(SegmentationS3DISSphere):

    INSTANCE_CLASSES = STUFF_CLASSES_INV.keys()
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
        """ Adds extra labels for the instance and object segmentation tasks
        - num_instances: number of instances
        - center_label: [64, 3] on centre per instance
        - instance_labels: [num_points]
        - vote_label: [num_points, 3] displacmenet between each point and the center.
        - instance_mask: [num_points] boolean mask 
        """
        # Initaliase variables
        num_points = data.pos.shape[0]
        semantic_labels = data.y

        # compute votes *AFTER* augmentation
        instances = np.unique(data.instance_labels)
        centers = []
        point_votes = torch.zeros([num_points, 3])
        instance_labels = torch.zeros(num_points, dtype=torch.long)
        instance_idx = 1
        for i_instance in instances:
            # find all points belong to that instance
            ind = np.where(data.instance_labels == i_instance)[0]
            # find the semantic label
            instance_class = semantic_labels[ind[0]].item()
            if instance_class in self.INSTANCE_CLASSES:  # We keep this instance
                pos = data.pos[ind, :3]
                max_pox = pos.max(0)[0]
                min_pos = pos.min(0)[0]
                center = 0.5 * (min_pos + max_pox)
                point_votes[ind, :] = center - pos
                centers.append(torch.tensor(center))
                instance_labels[ind] = instance_idx
                instance_idx += 1

        num_instances = len(centers)
        if num_instances > self.NUM_MAX_OBJECTS:
            raise ValueError(
                "We have more objects than expected. Please increase the NUM_MAX_OBJECTS variable.")
        data.center_label = torch.zeros((self.NUM_MAX_OBJECTS, 3))
        if num_instances:
            data.center_label[:num_instances, :] = torch.stack(centers)

        data.vote_label = point_votes.float()
        data.instance_labels = instance_labels
        data.instance_mask = instance_labels != 0
        data.num_instances = torch.tensor([num_instances])

        # Remap labels
        data.y = self._remap_labels(data.y)
        return data

    def _remap_labels(self, semantic_label):
        return semantic_label

    @property
    def stuff_classes(self):
        return super()._remap_labels(self.INSTANCE_CLASSES)

    def process(self):
        super().process()

    def download(self):
        super().download()

class SegmentationS3DISCylinder(SegmentationS3DISSphere):
    def _get_random(self):
        # Random spheres biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        area_data = self._datas[centre[3].int()]
        cylinder_sampler = cT.CylinderSampling(self._radius, centre[:3], align_origin=False)
        return cylinder_sampler(area_data)

    def _load_data(self, path):
        self._datas = torch.load(path)
        if not isinstance(self._datas, list):
            self._datas = [self._datas]
        if self._sample_per_epoch > 0:
            self._centres_for_sampling = []
            for i, data in enumerate(self._datas):
                assert not hasattr(
                    data, cT.CylinderSampling.KDTREE_KEY
                )  # Just to make we don't have some out of date data in there
                low_res = self._grid_sphere_sampling(data.clone())
                centres = torch.empty(
                    (low_res.pos.shape[0], 5), dtype=torch.float)
                centres[:, :3] = low_res.pos
                centres[:, 3] = i
                centres[:, 4] = low_res.y
                self._centres_for_sampling.append(centres)
                tree = KDTree(np.asarray(data.pos[:, :-1]), leaf_size=10)
                setattr(data, cT.CylinderSampling.KDTREE_KEY, tree)

            self._centres_for_sampling = torch.cat(
                self._centres_for_sampling, 0)
            uni, uni_counts = np.unique(np.asarray(
                self._centres_for_sampling[:, -1]), return_counts=True)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            self._labels = uni
        else:
            grid_sampler = cT.GridCylinderSampling(
                self._radius, self._radius, center=False)
            self._test_spheres = grid_sampler(self._datas)   

class PanopticS3DISCylinder(SegmentationS3DISCylinder):

    INSTANCE_CLASSES = STUFF_CLASSES_INV.keys()
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
        """ Adds extra labels for the instance and object segmentation tasks
        - num_instances: number of instances
        - center_label: [64, 3] on centre per instance
        - instance_labels: [num_points]
        - vote_label: [num_points, 3] displacmenet between each point and the center.
        - instance_mask: [num_points] boolean mask 
        """
        # Initaliase variables
        num_points = data.pos.shape[0]
        semantic_labels = data.y

        # compute votes *AFTER* augmentation
        instances = np.unique(data.instance_labels)
        centers = []
        point_votes = torch.zeros([num_points, 3])
        instance_labels = torch.zeros(num_points, dtype=torch.long)
        instance_idx = 1
        for i_instance in instances:
            # find all points belong to that instance
            ind = np.where(data.instance_labels == i_instance)[0]
            # find the semantic label
            instance_class = semantic_labels[ind[0]].item()
            if instance_class in self.INSTANCE_CLASSES:  # We keep this instance
                pos = data.pos[ind, :3]
                max_pox = pos.max(0)[0]
                min_pos = pos.min(0)[0]
                center = 0.5 * (min_pos + max_pox)
                point_votes[ind, :] = center - pos
                centers.append(torch.tensor(center))
                instance_labels[ind] = instance_idx
                instance_idx += 1

        num_instances = len(centers)
        if num_instances > self.NUM_MAX_OBJECTS:
            raise ValueError(
                "We have more objects than expected. Please increase the NUM_MAX_OBJECTS variable.")
        data.center_label = torch.zeros((self.NUM_MAX_OBJECTS, 3))
        if num_instances:
            data.center_label[:num_instances, :] = torch.stack(centers)

        data.vote_label = point_votes.float()
        data.instance_labels = instance_labels
        data.instance_mask = instance_labels != 0
        data.num_instances = torch.tensor([num_instances])

        # Remap labels
        data.y = self._remap_labels(data.y)
        return data

    def _remap_labels(self, semantic_label):
        return semantic_label

    @property
    def stuff_classes(self):
        return super()._remap_labels(self.INSTANCE_CLASSES)

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

        sampling_format = dataset_opt.get('sampling_format', 'cylinder')
        dataset_cls = PanopticS3DISCylinder if sampling_format == 'cylinder' else PanopticS3DISSphere

        self.train_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=3000,
            test_area=self.dataset_opt.fold,
            split="train",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.train_transform,
            keep_instance=True
        )

        self.val_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="val",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.val_transform,
            keep_instance=True
        )
        self.test_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="test",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.test_transform,
            keep_instance=True
        )

        if dataset_opt.class_weight_method:
            self.train_dataset = add_weights(
                self.train_dataset, True, dataset_opt.class_weight_method)

    @property
    def test_data(self):
        return self.test_dataset[0].raw_test_data

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """

        return PanopticTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)


