import os
import numpy as np
import torch
import logging
from torch_geometric.data import InMemoryDataset
from torch_points3d.datasets.segmentation.scannet import Scannet
from torch_points3d.metrics.panoptic_tracker import PanopticTracker
from torch_points3d.datasets.base_dataset import BaseDataset, save_used_properties

DIR = os.path.dirname(os.path.realpath(__file__))

log = logging.getLogger(__name__)


class ScannetPanoptic(Scannet):
    NYU40IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                         14, 16, 24, 28, 33, 34, 36, 39])
    NUM_MAX_OBJECTS = 64

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.STUFFCLASSES = torch.tensor(
            [i for i in self.VALID_CLASS_IDS if i not in self.NYU40IDS])
        self.NYU40ID2CLASS = {nyu40id: i for i,
                              nyu40id in enumerate(list(self.NYU40IDS))}

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
            if instance_class in self.NYU40IDS:  # We keep this instance
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
        data.y = super()._remap_labels(data.y)
        return data

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

        use_instance_labels: bool = dataset_opt.use_instance_labels
        donotcare_class_ids: [
        ] = dataset_opt.donotcare_class_ids if dataset_opt.donotcare_class_ids else []
        max_num_point: int = dataset_opt.max_num_point if dataset_opt.max_num_point != "None" else None
        is_test: bool = dataset_opt.is_test if dataset_opt.is_test is not None else False

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
