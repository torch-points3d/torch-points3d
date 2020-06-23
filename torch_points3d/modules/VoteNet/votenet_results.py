import torch
from torch_geometric.data import Data
import numpy as np
from typing import List

from torch_points3d.core.losses.huber_loss import nn_distance
from torch_points3d.utils.box_utils import box_corners_from_param, nms_samecls
from torch_points3d.datasets.object_detection.box_data import BoxData


class VoteNetResults(Data):
    @classmethod
    def from_logits(
        cls,
        seed_inds: torch.Tensor,
        seed_votes: torch.Tensor,
        seed_pos: torch.Tensor,
        sampled_votes: torch.Tensor,
        features: torch.Tensor,
        num_classes: int,
        num_heading_bin: int,
        mean_size_arr: torch.Tensor,
    ):
        """ Takes the sampled votes and the output features from the proposal network to generate a structured data object with
        all necessary info for loss and metric computations

        Parameters
        ----------
        seed_inds: torch.tensor
            Index of the points that were selected as seeds
        seed_votes: torch.tensor
            All seed votes before sampling and aggregation
        seed_pos: torch.Tensor
            All seed points
        sampled_votes: torch.tensor
            Votes selected as support points for the proposal network
        features: torch.Tensor
            Output features of the proposal network
        num_classes: int
            Number of classes to predict
        num_heading_bin: int
            Number of bins for heading computations
        mean_size_arr: torch.Tensor
            Average size of the box per class in each direction

        Returns
        -------
        data:
            - seed_inds
            - seed_votes
            - seed_pos
            - sampled_votes - proposed centre of the box before aggregation [B,N,3]
            - objectness_scores - [B,N,2]
            - center - corrected centre of the box [B,N,3]
            - heading_scores - [B, N, num_heading_bin]
            - heading_residuals_normalized - between -1 and 1 [B, N, num_heading_bin]
            - heading_residual - between -PI and PI [B, N, num_heading_bin]
            - size_scores - [B,N,num_size_cluster]
            - size_residuals_normalized - [B,N,num_size_cluster, 3]
            - size_residuals - [B,N,num_size_cluster, 3]
            - sem_cls_scores - [B,N,num_classes]
        """
        num_size_cluster = len(mean_size_arr)
        assert features.dim() == 3
        assert features.shape[1] == 2 + 3 + num_heading_bin * 2 + num_size_cluster * 4 + num_classes

        data = cls(sampled_votes=sampled_votes, seed_inds=seed_inds, seed_votes=seed_votes, seed_pos=seed_pos)

        x_transposed = features.transpose(2, 1)  # (batch_size, num_proposal, features)
        batch_size = x_transposed.shape[0]
        num_proposal = x_transposed.shape[1]

        objectness_scores = x_transposed[:, :, 0:2]
        data.objectness_scores = objectness_scores

        base_xyz = sampled_votes  # (batch_size, num_proposal, 3)
        center = base_xyz + x_transposed[:, :, 2:5]  # (batch_size, num_proposal, 3)
        data.center = center

        heading_scores = x_transposed[:, :, 5 : 5 + num_heading_bin]
        heading_residuals_normalized = x_transposed[:, :, 5 + num_heading_bin : 5 + num_heading_bin * 2]
        data.heading_scores = heading_scores  # Bxnum_proposalxnum_heading_bin
        data.heading_residuals_normalized = (
            heading_residuals_normalized  # Bxnum_proposalxnum_heading_bin (should be -1 to 1) TODO check that!
        )
        data.heading_residuals = heading_residuals_normalized * (
            np.pi / num_heading_bin
        )  # Bxnum_proposalxnum_heading_bin

        size_scores = x_transposed[:, :, 5 + num_heading_bin * 2 : 5 + num_heading_bin * 2 + num_size_cluster]
        size_residuals_normalized = x_transposed[
            :, :, 5 + num_heading_bin * 2 + num_size_cluster : 5 + num_heading_bin * 2 + num_size_cluster * 4
        ].view(
            [batch_size, num_proposal, num_size_cluster, 3]
        )  # Bxnum_proposalxnum_size_clusterx3
        data.size_scores = size_scores
        data.size_residuals_normalized = size_residuals_normalized
        if len(mean_size_arr) > 0:
            data.size_residuals = size_residuals_normalized * mean_size_arr.unsqueeze(0).unsqueeze(0)
        sem_cls_scores = x_transposed[:, :, 5 + num_heading_bin * 2 + num_size_cluster * 4 :]  # Bxnum_proposalx10
        data.sem_cls_scores = sem_cls_scores

        return data

    def assign_objects(self, gt_center: torch.Tensor, near_threshold: float, far_threshold: float):
        """ Assigns an object to each prediction based on the closest ground truth
        objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
        objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
        object_assignment: Tensor with long int within [0,num_gt_object-1]

        Parameters
        ----------
        gt_center : torch.Tensor
            center of ground truth objects [B,K,3]
        near_threshold: float
        far_threshold: float
        """

        B = gt_center.shape[0]
        K = self.sampled_votes.shape[1]
        gt_center.shape[1]
        dist1, ind1, _, _ = nn_distance(
            self.sampled_votes, gt_center
        )  # dist1: BxK, dist2: BxK2 TODO Optimise this nn_distance function, does a lot of useless stuff
        # TODO Why computing the closest GT using the vote instead of the corrected center

        euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
        self.objectness_label = torch.zeros((B, K), dtype=torch.long).to(self.sampled_votes.device)
        self.objectness_mask = torch.zeros((B, K)).to(self.sampled_votes.device)
        self.objectness_label[euclidean_dist1 < near_threshold] = 1
        self.objectness_mask[euclidean_dist1 < near_threshold] = 1
        self.objectness_mask[euclidean_dist1 > far_threshold] = 1

        self.object_assignment = ind1

    @property
    def batch_size(self):
        return self.center.shape[0]

    @property
    def num_proposal(self):
        return self.center.shape[1]

    def get_boxes(
        self, dataset, apply_nms=False, objectness_threshold=0.05, duplicate_boxes=False
    ) -> List[List[BoxData]]:
        """ Generates boxes from predictions

        Parameters
        ----------
        dataset :
            Must provide a class2size method and a class2angle method that return the angle and size
            for a given object class and residual value
        apply_nms: bool
            If True then we apply non max suppression before returning the boxes
        duplicate_boxes: bool
            If True then we duplicate predicted boxes accross all classes. Else we assign the box to the
            most likely class

        Returns
        -------
        List[List[BoxData]] contains the list of predicted boxes for each batch
        """

        # Size and Heading prediciton
        pred_heading_class = torch.argmax(self.heading_scores, -1)  # B,num_proposal
        pred_heading_residual = torch.gather(
            self.heading_residuals, 2, pred_heading_class.unsqueeze(-1)
        )  # B,num_proposal,1
        pred_size_class = torch.argmax(self.size_scores, -1)  # B,num_proposal
        pred_size_residual = torch.gather(
            self.size_residuals, 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3)
        ).squeeze(
            2
        )  # B,num_proposal,3

        # Generate box corners
        pred_corners_3d = torch.zeros((self.batch_size, self.num_proposal, 8, 3))
        for i in range(self.batch_size):
            for j in range(self.num_proposal):
                heading_angle = dataset.class2angle(pred_heading_class[i, j], pred_heading_residual[i, j])
                box_size = dataset.class2size(pred_size_class[i, j], pred_size_residual[i, j])
                corners_3d = box_corners_from_param(box_size, heading_angle, self.center[i, j, :])
                pred_corners_3d[i, j] = corners_3d

        # Objectness and class
        pred_obj = torch.nn.functional.softmax(self.objectness_scores, -1)[:, :, 1]  # B,num_proposal
        pred_sem_cls = torch.argmax(self.sem_cls_scores, -1)  # B,num_proposal

        # Apply nms if required
        if apply_nms:
            mask = self._nms_mask(pred_corners_3d, pred_obj, pred_sem_cls)
        else:
            mask = np.ones((self.batch_size, self.num_proposal), dtype=np.bool)

        detected_boxes = []
        sem_cls_proba = torch.softmax(self.sem_cls_scores, -1)
        for i in range(self.batch_size):
            corners = pred_corners_3d[i, mask[i]]
            objectness = pred_obj[i, mask[i]]
            sem_cls_scores = sem_cls_proba[i, mask[i]]
            clsname = pred_sem_cls[i, mask[i]]

            # Build box data for each detected object and add it to the list
            batch_detection = []
            for j in range(len(corners)):
                if objectness[j] > objectness_threshold:
                    if duplicate_boxes:
                        for classname in range(self.sem_cls_scores.shape[-1]):
                            batch_detection.append(
                                BoxData(classname, corners[j], score=objectness[j] * sem_cls_scores[j, classname])
                            )
                    else:
                        batch_detection.append(BoxData(clsname[j], corners[j], score=objectness[j]))

            detected_boxes.append(batch_detection)

        return detected_boxes

    def _nms_mask(self, pred_corners_3d, objectness, pred_sem_cls):
        """
        Parameters
        ----------
        pred_corners_3d : [B, num_proposal, 8, 3]
            box corners
        objectness: [B, num_proposal]
            objectness score
        pred_sem_cls: [B, num_proposal]
            Predicted semantic class
        """
        boxes_3d = torch.zeros((self.batch_size, self.num_proposal, 6))  # [xmin, ymin, zmin, xmax, ymax, zmax]
        boxes_3d[:, :, 0] = torch.min(pred_corners_3d[:, :, :, 0], dim=2)[0]
        boxes_3d[:, :, 1] = torch.min(pred_corners_3d[:, :, :, 1], dim=2)[0]
        boxes_3d[:, :, 2] = torch.min(pred_corners_3d[:, :, :, 2], dim=2)[0]
        boxes_3d[:, :, 3] = torch.max(pred_corners_3d[:, :, :, 0], dim=2)[0]
        boxes_3d[:, :, 4] = torch.max(pred_corners_3d[:, :, :, 1], dim=2)[0]
        boxes_3d[:, :, 5] = torch.max(pred_corners_3d[:, :, :, 2], dim=2)[0]

        boxes_3d = boxes_3d.cpu().numpy()
        mask = np.zeros((self.batch_size, self.num_proposal), dtype=np.bool)
        for b in range(self.batch_size):
            pick = nms_samecls(boxes_3d[b], pred_sem_cls[b], objectness[b], overlap_threshold=0.25)
            mask[b, pick] = True
        return mask
