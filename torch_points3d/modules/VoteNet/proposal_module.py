""" Adapted from VoteNet

Ref: https://github.com/facebookresearch/votenet/blob/master/models/proposal_module.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch_points3d.modules.pointnet2 import PointNetMSGDown
import torch_points_kernels as tp
from .votenet_results import VoteNetResults


class ProposalModule(nn.Module):
    def __init__(
        self,
        num_class,
        vote_aggregation_config,
        num_heading_bin,
        mean_size_arr,
        num_proposal,
        sampling,
        seed_feat_dim=256,
    ):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = len(mean_size_arr)
        self.mean_size_arr = nn.Parameter(torch.Tensor(mean_size_arr), requires_grad=False)
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        assert (
            vote_aggregation_config.module_name == "PointNetMSGDown"
        ), "Proposal Module support only PointNet2 for now"
        params = OmegaConf.to_container(vote_aggregation_config)
        self.vote_aggregation = PointNetMSGDown(**params)
        pn2_output_nc = vote_aggregation_config.down_conv_nn[-1][-1]
        output_feat = 2 + 3 + num_heading_bin * 2 + self.num_size_cluster * 4 + self.num_class
        mid_feat = (pn2_output_nc + output_feat) // 2
        self.conv1 = torch.nn.Conv1d(pn2_output_nc, pn2_output_nc, 1)
        self.conv2 = torch.nn.Conv1d(pn2_output_nc, mid_feat, 1)
        self.conv3 = torch.nn.Conv1d(mid_feat, output_feat, 1)
        self.bn1 = torch.nn.BatchNorm1d(pn2_output_nc)
        self.bn2 = torch.nn.BatchNorm1d(mid_feat)

    def forward(self, data):
        """
        Args:
            pos: (B,N,3)
            features: (B,C,N)
            seed_pos (B,N,3)
        Returns:
            VoteNetResults
        """
        if data.pos.dim() != 3:
            raise ValueError("This method only supports dense convolutions for now")

        if self.sampling == "seed_fps":
            sample_idx = tp.furthest_point_sample(data.seed_pos, self.num_proposal)
        else:
            raise ValueError("Unknown sampling strategy: %s. Exiting!" % (self.sampling))

        data_features = self.vote_aggregation(data, sampled_idx=sample_idx)

        # --------- PROPOSAL GENERATION ---------
        x = F.relu(self.bn1(self.conv1(data_features.x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)  # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)

        return VoteNetResults.from_logits(
            data.seed_inds,
            data.pos,
            data.seed_pos,
            data_features.pos,
            x,
            self.num_class,
            self.num_heading_bin,
            self.mean_size_arr,
        )
