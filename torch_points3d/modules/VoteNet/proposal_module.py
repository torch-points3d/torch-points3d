# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf

from torch_points3d.modules.pointnet2 import PointNetMSGDown
import torch_points_kernels as tp


def decode_scores(data, x, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
    """ Returns a data object with:
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
    x_transposed = x.transpose(2, 1)  # (batch_size, num_proposal, features)
    batch_size = x_transposed.shape[0]
    num_proposal = x_transposed.shape[1]

    objectness_scores = x_transposed[:, :, 0:2]
    data.objectness_scores = objectness_scores

    base_xyz = data.aggregated_vote_xyz  # (batch_size, num_proposal, 3)
    center = base_xyz + x_transposed[:, :, 2:5]  # (batch_size, num_proposal, 3)
    data.center = center

    heading_scores = x_transposed[:, :, 5 : 5 + num_heading_bin]
    heading_residuals_normalized = x_transposed[:, :, 5 + num_heading_bin : 5 + num_heading_bin * 2]
    data.heading_scores = heading_scores  # Bxnum_proposalxnum_heading_bin
    data.heading_residuals_normalized = (
        heading_residuals_normalized  # Bxnum_proposalxnum_heading_bin (should be -1 to 1) TODO check that!
    )
    data.heading_residuals = heading_residuals_normalized * (np.pi / num_heading_bin)  # Bxnum_proposalxnum_heading_bin

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


class ProposalModule(nn.Module):
    def __init__(
        self,
        num_class,
        vote_aggregation_config,
        num_heading_bin,
        num_size_cluster,
        mean_size_arr,
        num_proposal,
        sampling,
        seed_feat_dim=256,
    ):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = nn.Parameter(torch.Tensor(mean_size_arr), requires_grad=False)
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        assert (
            vote_aggregation_config.module_name == "PointNetMSGDown"
        ), "Proposal Module support only PointNet2 for now"
        params = OmegaConf.to_container(vote_aggregation_config)
        self.vote_aggregation = PointNetMSGDown(**params)

        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(128, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 2 + 3 + num_heading_bin * 2 + num_size_cluster * 4 + self.num_class, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

    def forward(self, data):
        """
        Args:
            pos: (B,K,3)
            features: (B,C,K)
            seed_pos (B,N,3)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """
        if data.pos.dim() != 3:
            raise ValueError("This method only supports dense convolutions for now")
        if self.sampling == "seed_fps":
            sample_idx = tp.furthest_point_sample(data.seed_pos, self.num_proposal)
        else:
            raise ValueError("Unknown sampling strategy: %s. Exiting!" % (self.sampling))

        data_features = self.vote_aggregation(data, sampled_idx=sample_idx)
        data.aggregated_vote_xyz = data_features.pos  # (batch_size, num_proposal, 3)
        data.aggregated_vote_inds = sample_idx  # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        x = F.relu(self.bn1(self.conv1(data_features.x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)  # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)

        return decode_scores(data, x, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)
