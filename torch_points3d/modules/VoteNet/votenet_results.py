import torch
from torch_geometric.data import Data
import numpy as np

from torch_points3d.core.losses.huber_loss import nn_distance


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
        num_size_cluster: int,
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
        num_size_cluster: int
            Number of clusters for size computations
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
            centres of ground truth objects [B,K,3]
        near_threshold: float
        far_threshold: float
        """

        B = gt_center.shape[0]
        K = self.sampled_votes.shape[1]
        gt_center.shape[1]
        dist1, ind1, _, _ = nn_distance(
            self.sampled_votes, gt_center
        )  # dist1: BxK, dist2: BxK2 TODO Optimise this nn_distance function, does a lot of useless stuff
        # TODO Why computing the closest GT using the vote instead of the corrected centre

        euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
        self.objectness_label = torch.zeros((B, K), dtype=torch.long).to(self.sampled_votes.device)
        self.objectness_mask = torch.zeros((B, K)).to(self.sampled_votes.device)
        self.objectness_label[euclidean_dist1 < near_threshold] = 1
        self.objectness_mask[euclidean_dist1 < near_threshold] = 1
        self.objectness_mask[euclidean_dist1 > far_threshold] = 1

        self.object_assignment = ind1
