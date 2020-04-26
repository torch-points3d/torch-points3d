# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

from torch_points3d.modules.pointnet2 import PointNetMSGDown
import torch_points_kernels as tp


def decode_scores(net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
    net_transposed = net.transpose(2, 1)  # (batch_size, 1024, ..)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    objectness_scores = net_transposed[:, :, 0:2]
    end_points["objectness_scores"] = objectness_scores

    base_xyz = end_points["aggregated_vote_xyz"]  # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:, :, 2:5]  # (batch_size, num_proposal, 3)
    end_points["center"] = center

    heading_scores = net_transposed[:, :, 5 : 5 + num_heading_bin]
    heading_residuals_normalized = net_transposed[:, :, 5 + num_heading_bin : 5 + num_heading_bin * 2]
    end_points["heading_scores"] = heading_scores  # Bxnum_proposalxnum_heading_bin
    end_points[
        "heading_residuals_normalized"
    ] = heading_residuals_normalized  # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points["heading_residuals"] = heading_residuals_normalized * (
        np.pi / num_heading_bin
    )  # Bxnum_proposalxnum_heading_bin

    size_scores = net_transposed[:, :, 5 + num_heading_bin * 2 : 5 + num_heading_bin * 2 + num_size_cluster]
    size_residuals_normalized = net_transposed[
        :, :, 5 + num_heading_bin * 2 + num_size_cluster : 5 + num_heading_bin * 2 + num_size_cluster * 4
    ].view(
        [batch_size, num_proposal, num_size_cluster, 3]
    )  # Bxnum_proposalxnum_size_clusterx3
    end_points["size_scores"] = size_scores
    end_points["size_residuals_normalized"] = size_residuals_normalized
    end_points["size_residuals"] = size_residuals_normalized * torch.from_numpy(
        mean_size_arr.astype(np.float32)
    ).cuda().unsqueeze(0).unsqueeze(0)

    sem_cls_scores = net_transposed[:, :, 5 + num_heading_bin * 2 + num_size_cluster * 4 :]  # Bxnum_proposalx10
    end_points["sem_cls_scores"] = sem_cls_scores
    return end_points


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
        import pdb

        pdb.set_trace()
        self.vote_aggregation = PointNetMSGDown(**params)

        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(128, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 2 + 3 + num_heading_bin * 2 + num_size_cluster * 4 + self.num_class, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

    def forward(self, data_votes):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """
        if self.sampling == "vote_fps":
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling == "seed_fps":
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = tp.furthest_point_sample(end_points["seed_xyz"], self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == "random":
            # Random sampling from the votes
            num_seed = end_points["seed_xyz"].shape[1]
            batch_size = end_points["seed_xyz"].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            log_string("Unknown sampling strategy: %s. Exiting!" % (self.sampling))
            exit()
        end_points["aggregated_vote_xyz"] = xyz  # (batch_size, num_proposal, 3)
        end_points[
            "aggregated_vote_inds"
        ] = sample_inds  # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        net = F.relu(self.bn1(self.conv1(features)))
        net = F.relu(self.bn2(self.conv2(net)))
        net = self.conv3(net)  # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)

        end_points = decode_scores(
            net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr
        )
        return end_points


if __name__ == "__main__":
    sys.path.append(os.path.join(ROOT_DIR, "sunrgbd"))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC

    net = ProposalModule(
        DC.num_class, DC.num_heading_bin, DC.num_size_cluster, DC.mean_size_arr, 128, "seed_fps"
    ).cuda()
    end_points = {"seed_xyz": torch.rand(8, 1024, 3).cuda()}
    out = net(torch.rand(8, 1024, 3).cuda(), torch.rand(8, 256, 1024).cuda(), end_points)
    for key in out:
        print(key, out[key].shape)
