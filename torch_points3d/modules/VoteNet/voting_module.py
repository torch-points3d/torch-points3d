"""
Adapted from VoteNet

Ref: https://github.com/facebookresearch/votenet/blob/master/models/voting_module.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


class VotingModule(nn.Module):
    def __init__(self, vote_factor, seed_feature_dim, conv_type="DENSE"):
        """ Votes generation from seed point features.

        Args:
            vote_facotr: int
                number of votes generated from each seed point
            seed_feature_dim: int
                number of channels of seed point features
            vote_feature_dim: int
                number of channels of vote features
        """
        super().__init__()
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = self.in_dim  # due to residual feature, in_dim has to be == out_dim
        self._conv_type = conv_type
        if conv_type == "DENSE":
            self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
            self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
            self.conv3 = torch.nn.Conv1d(self.in_dim, (3 + self.out_dim) * self.vote_factor, 1)
            self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
            self.bn2 = torch.nn.BatchNorm1d(self.in_dim)
        else:
            self.conv1 = torch.nn.Linear(self.in_dim, self.in_dim)
            self.conv2 = torch.nn.Linear(self.in_dim, self.in_dim)
            self.conv3 = torch.nn.Linear(self.in_dim, (3 + self.out_dim) * self.vote_factor)
            self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
            self.bn2 = torch.nn.BatchNorm1d(self.in_dim)

    def forward(self, data):
        """ Votes for centres using a PN++ like architecture
        Returns
        -------
        data:
            - pos: position of the vote (centre of the box)
            - x: feature of the vote (original feature + processed feature)
            - seed_pos: position of the original point
        """
        if 3 < data.pos.dim() and data.pos.dim() <= 1:
            raise Exception("data.pos doesn t have the correct dimension. Should be either 2 or 3")

        if self._conv_type == "DENSE":
            batch_size = data.pos.shape[0]
            num_points = data.pos.shape[1]
            num_votes = num_points * self.vote_factor
            x = F.relu(self.bn1(self.conv1(data.x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.conv3(x)  # (batch_size, (3+out_dim)*vote_factor, num_seed)

            x = x.transpose(2, 1).view(batch_size, num_points, self.vote_factor, 3 + self.out_dim)
            offset = x[:, :, :, 0:3]
            vote_pos = data.pos.unsqueeze(2) + offset.contiguous()
            vote_pos = vote_pos.contiguous().view(batch_size, num_votes, 3)

            res_x = x[:, :, :, 3:]  # (batch_size, num_seed, vote_factor, out_dim)
            vote_x = data.x.transpose(2, 1).unsqueeze(2).contiguous() + res_x.contiguous()
            vote_x = vote_x.contiguous().view(batch_size, num_votes, self.out_dim)
            vote_x = vote_x.transpose(2, 1).contiguous()
            data_out = Data(pos=vote_pos, x=vote_x, seed_pos=data.pos)
        else:
            x = F.relu(self.bn1(self.conv1(data.x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.conv3(x)
            offset = x[:, 0:3]
            vote_pos = data.pos + offset
            res_x = x[:, 3:]
            vote_x = data.x + res_x
            data_out = Data(pos=vote_pos, x=vote_x, seed_pos=data.pos, batch=data.batch)
        return data_out


if __name__ == "__main__":
    net = VotingModule(2, 256)
    data_votes = net(Data(pos=torch.rand(8, 1024, 3), x=torch.rand(8, 256, 1024)))
    print("vote_pos", data_votes.pos.shape)
    print("vote_x", data_votes.x.shape)
