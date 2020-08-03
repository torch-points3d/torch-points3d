import os
import numpy as np
import torch
import logging
from torch_geometric.data import InMemoryDataset
from torch_points3d.datasets.segmentation.scannet import Scannet, NUM_CLASSES, IGNORE_LABEL
from torch_points3d.metrics.object_detection_tracker import ObjectDetectionTracker
from torch_points3d.datasets.base_dataset import BaseDataset, save_used_properties
from torch_points3d.utils.box_utils import box_corners_from_param

DIR = os.path.dirname(os.path.realpath(__file__))

log = logging.getLogger(__name__)


class ScannetObjectDetection(Scannet):

    MAX_NUM_OBJ = 64
    NUM_HEADING_BIN = 1
    TYPE2CLASS = {
        "cabinet": 0,
        "bed": 1,
        "chair": 2,
        "sofa": 3,
        "table": 4,
        "door": 5,
        "window": 6,
        "bookshelf": 7,
        "picture": 8,
        "counter": 9,
        "desk": 10,
        "curtain": 11,
        "refrigerator": 12,
        "showercurtrain": 13,
        "toilet": 14,
        "sink": 15,
        "bathtub": 16,
        "garbagebin": 17,
    }
    NYU40IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
    MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

    def __init__(self, *args, **kwargs):
        super(ScannetObjectDetection, self).__init__(*args, **kwargs)

        self.CLASS2TYPE = {self.TYPE2CLASS[t]: t for t in self.TYPE2CLASS}
        self.NYU40ID2CLASS = {nyu40id: i for i, nyu40id in enumerate(list(self.NYU40IDS))}
        self.MEAN_SIZE_ARR = np.load(os.path.join(DIR, "scannet_metadata/scannet_means.npz"))["arr_0"]
        self.TYPE_MEAN_SIZE = {}
        for i in range(len(self.NYU40IDS)):
            self.TYPE_MEAN_SIZE[self.CLASS2TYPE[i]] = self.MEAN_SIZE_ARR[i, :]

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
        instance_box_corners: (MAX_NUM_OBJ, 8, 3) corners of the bounding boxes in this room
        center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
        sem_cls_label: (MAX_NUM_OBJ,) semantic class index
        angle_residual_label: (MAX_NUM_OBJ,)
        size_residual_label: (MAX_NUM_OBJ,3)
        box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        vote_label: (N,3) with votes XYZ
        vote_label_mask: (N,) with 0/1 with 1 indicating the point is in one of the object's OBB.
        """
        # Initaliase variables
        num_points = data.pos.shape[0]
        semantic_labels = data.y
        instance_labels = data.instance_labels

        center_label = torch.zeros((self.MAX_NUM_OBJ, 3))
        target_bboxes_mask = torch.zeros((self.MAX_NUM_OBJ), dtype=torch.bool)
        angle_residuals = torch.zeros((self.MAX_NUM_OBJ,))
        size_classes = torch.zeros((self.MAX_NUM_OBJ,))
        size_residuals = torch.zeros((self.MAX_NUM_OBJ, 3))

        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label.
        point_votes = torch.zeros([num_points, 3])
        point_votes_mask = torch.zeros(num_points, dtype=torch.bool)
        instance_box_corners = []
        box_sizes = []
        centers = []
        instance_classes = []
        for i_instance in np.unique(instance_labels):
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label
            instance_class = semantic_labels[ind[0]].item()
            if instance_class in self.NYU40IDS:
                pos = data.pos[ind, :3]
                max_pox = pos.max(0)[0]
                min_pos = pos.min(0)[0]
                center = 0.5 * (min_pos + max_pox)
                point_votes[ind, :] = center - pos
                point_votes_mask[ind] = True
                box_size = max_pox - min_pos
                instance_box_corners.append(box_corners_from_param(box_size, 0, center))
                box_sizes.append(box_size)
                centers.append(center)
                instance_classes.append(self.NYU40ID2CLASS[instance_class])
        point_votes = point_votes.repeat((1, 3))  # make 3 votes identical
        instance_classes = torch.tensor(instance_classes)

        # Keep only boxes with valid ids
        num_instances = len(instance_classes)
        target_bboxes_mask[0:num_instances] = True

        # Set box semantic label
        target_bboxes_semcls = np.zeros((self.MAX_NUM_OBJ))
        target_bboxes_semcls[0:num_instances] = instance_classes

        # Set size residual and box centres
        size_classes[0:num_instances] = instance_classes
        if num_instances > 0:
            box_sizes = torch.stack(box_sizes)
            centers = torch.stack(centers)
            size_residuals[0:num_instances, :] = box_sizes - torch.from_numpy(self.MEAN_SIZE_ARR[instance_classes, :])
            center_label[0:num_instances, :] = centers

        data.center_label = center_label
        data.heading_class_label = torch.zeros((self.MAX_NUM_OBJ,))
        data.heading_residual_label = angle_residuals.float()
        data.size_class_label = size_classes
        data.size_residual_label = size_residuals.float()
        data.sem_cls_label = torch.from_numpy(target_bboxes_semcls).int()
        data.box_label_mask = target_bboxes_mask
        data.vote_label = point_votes.float()
        data.vote_label_mask = point_votes_mask
        data.instance_box_corners = torch.zeros((self.MAX_NUM_OBJ, 8, 3))
        if len(instance_box_corners):
            data.instance_box_corners[: len(instance_box_corners), :, :] = torch.stack(instance_box_corners)

        delattr(data, "instance_bboxes")
        delattr(data, "instance_labels")

        # Remap labels
        data.y = super()._remap_labels(data.y)
        return data

    def _remap_labels(self, semantic_label):
        return semantic_label

    def process(self):
        super().process()

    def download(self):
        super().download()


class ScannetDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        use_instance_labels: bool = dataset_opt.use_instance_labels
        use_instance_bboxes: bool = dataset_opt.use_instance_bboxes
        donotcare_class_ids: [] = dataset_opt.donotcare_class_ids if dataset_opt.donotcare_class_ids else []
        max_num_point: int = dataset_opt.max_num_point if dataset_opt.max_num_point != "None" else None
        is_test: bool = dataset_opt.is_test if dataset_opt.is_test is not None else False

        self.train_dataset = ScannetObjectDetection(
            self._data_path,
            split="train",
            pre_transform=self.pre_transform,
            transform=self.train_transform,
            version=dataset_opt.version,
            use_instance_labels=use_instance_labels,
            use_instance_bboxes=use_instance_bboxes,
            donotcare_class_ids=donotcare_class_ids,
            max_num_point=max_num_point,
            is_test=is_test,
        )

        self.val_dataset = ScannetObjectDetection(
            self._data_path,
            split="val",
            transform=self.val_transform,
            pre_transform=self.pre_transform,
            version=dataset_opt.version,
            use_instance_labels=use_instance_labels,
            use_instance_bboxes=use_instance_bboxes,
            donotcare_class_ids=donotcare_class_ids,
            max_num_point=max_num_point,
            is_test=is_test,
        )

    @property  # type: ignore
    @save_used_properties
    def mean_size_arr(self):
        return self.train_dataset.MEAN_SIZE_ARR.copy()

    def class2angle(self, pred_cls, residual, to_label_format=True):
        """ Inverse function to angle2class.
        As ScanNet only has axis-alined boxes so angles are always 0. """
        return 0

    def class2size(self, pred_cls, residual):
        """ Inverse function to size2class """
        if torch.is_tensor(residual):
            mean = torch.tensor(self.mean_size_arr[pred_cls, :]).to(residual.device)
        else:
            mean = self.mean_size_arr[pred_cls, :]
        return mean + residual

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
        Returns:
            [BaseTracker] -- tracker
        """
        return ObjectDetectionTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
