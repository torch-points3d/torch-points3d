import numpy as np
import torch
from torch_points3d.datasets.segmentation.s3dis import S3DISFusedDataset as SegmentationS3DISFusedDataset

INV_OBJECT_LABEL = {
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

class S3DISFusedDataset(SegmentationS3DISFusedDataset):

    STUFFCLASSES = INV_OBJECT_LABEL.keys()
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
            if instance_class in self.STUFFCLASSES:  # We keep this instance
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



