import torch
import MinkowskiEngine as ME
from typing import List
from torch_geometric.data import Data

from torch_points3d.utils.box_utils import box3d_iou_aligned


class GSDNResult(Data):
    """
    centres: torch.Tensor = None  # [N, len(anchors), 3]
    centre_logits: torch.Tensor = None  # [N*len(anchors), 3]
    sizes: torch.Tensor = None  # [N,len(anchors),3]
    size_logits: torch.Tensor = None  # [N*len(anchors) , 3]
    class_logits: torch.Tensor = None  # [N*len(anchors),num_classes]
    objectness: torch.Tensor = None  # [N,len(anchors)]
    sparsity [N]
    batch: torch.Tensor = None  # [N]
    grid_coords: torch.Tensor = None
    grid_size: float = None
    anchors: torch.Tensor = None

    # For training only
    positive_mask: torch.Tensor = None  # [N*nb_anchors]
    negative_mask: torch.Tensor = None  # [N*nb_anchors]
    class_labels: torch.Tensor = None  # [N*nb_anchors]
    centre_labels: torch.Tensor = None  # [N*nb_anchors, 3]
    size_labels: torch.Tensor = None  # [N*nb_anchors,3 ]
    sparsity_positive: torch.Tensor = None  # [N]
    sparsity_negative: torch.Tensor = None  # [N]
    """

    @classmethod
    def create_from_logits(cls, sparse_tensor, box_logits, nb_anchors, sparsity):
        """ Generates GSDN results based on the output of a generative layer
        """
        coords = sparse_tensor.C
        assert box_logits.shape[0] == coords.shape[0]
        assert box_logits.shape[1] % nb_anchors == 0

        centre_logits = box_logits[:, 0 : nb_anchors * 3].reshape(-1, 3)
        size_logits = box_logits[:, nb_anchors * 3 : nb_anchors * 6].reshape(-1, 3)

        objectness_logits = box_logits[:, nb_anchors * 6 : nb_anchors * 7]
        class_logits = box_logits[:, nb_anchors * 7 :].reshape(coords.shape[0] * nb_anchors, -1)

        return cls(
            centre_logits=centre_logits,
            size_logits=size_logits,
            objectness=objectness_logits,
            class_logits=class_logits,
            batch=coords[:, 0].to(centre_logits.device),
            grid_coords=coords[:, 1:].to(centre_logits.device),
            sparsity=sparsity,
            coords_key=sparse_tensor.coords_key,
        )

    def evaluate_labels(
        self,
        centre_labels: List[torch.Tensor],
        size_labels: List[torch.Tensor],
        class_labels: List[torch.Tensor],
        negative_iou=0.2,
        positive_iou=0.35,
    ):
        """ Computes positive and negative boxes and associate ground truth boxes (size, semantic class and centre)
        to the positive predictions
        """
        self._init_labels()
        previous_idx = 0
        nb_anchors = len(self.anchors)

        for i in range(len(centre_labels)):
            # Get centres and sizes for all anchors in that sample
            centres, sizes = self._get_for_sample(i)
            centres = centres.reshape(-1, 3)  # [N*nb_anchors, 3]
            sizes = sizes.reshape(-1, 3)

            if len(centre_labels[i]) == 0:
                continue

            # Compute best matching box
            ious = box3d_iou_aligned(centres, sizes, centre_labels[i], size_labels[i])  # [N*nb_anchors, n_gt_boxes]
            max_ious, max_ious_ind = torch.max(ious, dim=-1)

            # Set labels
            nb_boxes_in_sample = centres.shape[0]
            self.positive_mask[previous_idx : previous_idx + nb_boxes_in_sample] = max_ious > positive_iou
            self.negative_mask[previous_idx : previous_idx + nb_boxes_in_sample] = max_ious < negative_iou

            self.class_labels[previous_idx : previous_idx + nb_boxes_in_sample] = class_labels[i][max_ious_ind]
            self.size_labels[previous_idx : previous_idx + nb_boxes_in_sample] = size_labels[i][max_ious_ind]
            self.centre_labels[previous_idx : previous_idx + nb_boxes_in_sample] = centre_labels[i][max_ious_ind]

            # increment sample idx
            previous_idx += nb_boxes_in_sample

        # Set sparsity labels
        self.sparsity_positive = (
            torch.sum(self.positive_mask.reshape(-1, nb_anchors), -1) > 0
        )  # at least one positive anchor
        self.sparsity_negative = (
            torch.sum(self.negative_mask.reshape(-1, nb_anchors), -1) == nb_anchors
        )  # all anchors negative

    def _get_for_sample(self, sample_id):
        mask = self.batch == sample_id
        if not hasattr(self, "centres") or self.centres is None:
            self._compute_box_predictions()

        return self.centres[mask], self.sizes[mask]

    def _compute_box_predictions(self):
        assert self.anchors is not None
        assert self.grid_size is not None

        nb_anchors = len(self.anchors)
        nb_boxes = self.grid_coords.shape[0]

        anchor_sizes = self.anchors.unsqueeze(0).repeat(nb_boxes, 1, 1)  # [N, len(anchors), 3]

        centre_coord = (
            (self.grid_coords.float() * self.grid_size).unsqueeze(1).repeat(1, nb_anchors, 1)
        )  # [N, len(anchors), 3]
        corrected_centres = centre_coord + self.centre_logits.reshape(nb_boxes, nb_anchors, 3) * anchor_sizes
        corrected_sizes = torch.exp(self.size_logits.reshape(nb_boxes, nb_anchors, 3)) * anchor_sizes

        self.centres = corrected_centres
        self.sizes = corrected_sizes

    def get_rescaled_centre_labels(self):
        """ Takes the centre label for each predicted box and rescales it to the scale of the logits
        see Faster-RCNN rescaling strategy https://arxiv.org/abs/1506.01497
        """
        nb_anchors = len(self.anchors)
        nb_boxes = self.grid_coords.shape[0]
        centre_labels = self.centre_labels.reshape(-1, nb_anchors, 3)
        anchor_sizes = self.anchors.unsqueeze(0).repeat(nb_boxes, 1, 1)  # [N, len(anchors), 3]
        anchor_centre = (
            (self.grid_coords.float() * self.grid_size).unsqueeze(1).repeat(1, nb_anchors, 1)
        )  # [N, len(anchors), 3]
        corrected_centre = (centre_labels - anchor_centre) / anchor_sizes
        return corrected_centre.reshape(-1, 3)

    def get_rescaled_size_labels(self):
        """ Takes the box size ground tructh for each predicted box and rescales it to the scale of the logits
        see Faster-RCNN rescaling strategy https://arxiv.org/abs/1506.01497
        """
        nb_anchors = len(self.anchors)
        nb_boxes = self.grid_coords.shape[0]
        size_labels = self.size_labels.reshape(-1, nb_anchors, 3)
        anchor_sizes = self.anchors.unsqueeze(0).repeat(nb_boxes, 1, 1)  # [N, len(anchors), 3]
        corrected_size = torch.log(size_labels / anchor_sizes)
        return corrected_size.reshape(-1, 3)

    def _init_labels(self):
        nb_anchors = len(self.anchors)
        self.centre_labels = torch.empty((nb_anchors * self.grid_coords.shape[0], 3), dtype=torch.float).to(self.device)
        self.size_labels = torch.empty((nb_anchors * self.grid_coords.shape[0], 3), dtype=torch.float).to(self.device)
        self.class_labels = torch.empty((nb_anchors * self.grid_coords.shape[0]), dtype=torch.long).to(self.device)
        self.positive_mask = torch.zeros((nb_anchors * self.grid_coords.shape[0]), dtype=torch.bool).to(self.device)
        self.negative_mask = torch.zeros((nb_anchors * self.grid_coords.shape[0]), dtype=torch.bool).to(self.device)

    @property
    def device(self):
        return self.centre_logits.device

    def set_sparsity(self, child_box, coords_manager):
        """ Updates the current sparsity label based on the next layer sparsity label
        """
        assert abs(self.coords_key.getTensorStride()[0] / child_box.coords_key.getTensorStride()[0] - 2) < 1e-5

        child_valid_coords = coords_manager.get_coords(child_box.coords_key)[child_box.sparsity_positive]
        if child_valid_coords.shape[0] == 0:
            return

        child_sparse_tensor = ME.SparseTensor(
            feats=torch.ones((child_valid_coords.shape[0], 1)).float(),
            coords=child_valid_coords,
            coords_manager=coords_manager,
            force_creation=True,
        )
        strided_child = coords_manager.stride(child_sparse_tensor.coords_key, 2, force_creation=True)
        child, parents = coords_manager.get_kernel_map(strided_child, self.coords_key, kernel_size=1)

        # Update masks
        for p in parents:
            self.sparsity_positive[p] = True
            self.sparsity_negative[p] = False
