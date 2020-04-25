# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "utils"))
sys.path.append(os.path.join(ROOT_DIR, "pointnet2"))

from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule


class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network.

       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """

    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
            npoint=2048, radius=0.2, nsample=64, mlp=[input_feature_dim, 64, 64, 128], use_xyz=True, normalize_xyz=True
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=1024, radius=0.4, nsample=32, mlp=[128, 128, 128, 256], use_xyz=True, normalize_xyz=True
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=512, radius=0.8, nsample=16, mlp=[256, 128, 128, 256], use_xyz=True, normalize_xyz=True
        )

        self.sa4 = PointnetSAModuleVotes(
            npoint=256, radius=1.2, nsample=16, mlp=[256, 128, 128, 256], use_xyz=True, normalize_xyz=True
        )

        self.fp1 = PointnetFPModule(mlp=[256 + 256, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 256, 256, 256])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points:
            end_points = {}
        pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points["sa1_inds"] = fps_inds
        end_points["sa1_xyz"] = xyz
        end_points["sa1_features"] = features

        xyz, features, fps_inds = self.sa2(xyz, features)  # this fps_inds is just 0,1,...,1023
        end_points["sa2_inds"] = fps_inds
        end_points["sa2_xyz"] = xyz
        end_points["sa2_features"] = features

        xyz, features, fps_inds = self.sa3(xyz, features)  # this fps_inds is just 0,1,...,511
        end_points["sa3_xyz"] = xyz
        end_points["sa3_features"] = features

        xyz, features, fps_inds = self.sa4(xyz, features)  # this fps_inds is just 0,1,...,255
        end_points["sa4_xyz"] = xyz
        end_points["sa4_features"] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(
            end_points["sa3_xyz"], end_points["sa4_xyz"], end_points["sa3_features"], end_points["sa4_features"]
        )
        features = self.fp2(end_points["sa2_xyz"], end_points["sa3_xyz"], end_points["sa2_features"], features)
        end_points["fp2_features"] = features
        end_points["fp2_xyz"] = end_points["sa2_xyz"]
        num_seed = end_points["fp2_xyz"].shape[1]
        end_points["fp2_inds"] = end_points["sa1_inds"][:, 0:num_seed]  # indices among the entire input point clouds
        return end_points


if __name__ == "__main__":
    backbone_net = Pointnet2Backbone(input_feature_dim=3).cuda()
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(16, 20000, 6).cuda())
    for key in sorted(out.keys()):
        print(key, "\t", out[key].shape)
