# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_points3d.modules.pointnet2 import PointNetMSGDown
import torch_points_kernels as tp


def decode_scores(data, x, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
    x_transposed = x.transpose(2, 1)  # (batch_size, 1024, ..)
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
        heading_residuals_normalized  # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
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
    data.size_residuals = size_residuals_normalized * torch.from_numpy(
        mean_size_arr.astype(np.float32)
    ).cuda().unsqueeze(0).unsqueeze(0)

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
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        assert (
            vote_aggregation_config.module_name == "PointNetMSGDown"
        ), "Proposal Module support only PointNet2 for now"
        params = vote_aggregation_config.to_container()
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
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """
        if self.sampling == "seed_fps":
            data.idx = tp.furthest_point_sample(data.seed_pos, self.num_proposal)
            data_features = self.vote_aggregation(data)
        else:
            print("Unknown sampling strategy: %s. Exiting!" % (self.sampling))
            exit()

        data.aggregated_vote_xyz = data_features.pos  # (batch_size, num_proposal, 3)
        data.aggregated_vote_inds = data.idx  # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        x = F.relu(self.bn1(self.conv1(data_features.x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)  # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)

        return decode_scores(data, x, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)
