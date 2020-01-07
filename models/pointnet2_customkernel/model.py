from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import etw_pytorch_utils as pt_utils
from collections import namedtuple

import os
import sys
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, '../..'))
from utils.timer import time_func
from models.pointnet2_customkernel.modules import PointnetFPModule, PointnetSAModuleMSG


def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ["preds", "loss", "acc"])

    def model_fn(model, data, epoch=0, eval=False):
        with torch.set_grad_enabled(not eval):
            inputs, labels = data
            inputs = inputs.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)

            preds = model(inputs)
            loss = criterion(preds.view(labels.numel(), -1), labels.view(-1))

            _, classes = torch.max(preds, -1)
            acc = (classes == labels).float().sum() / labels.numel()

        return ModelReturn(preds, loss, {"acc": acc.item(), "loss": loss.item()})

    return model_fn


class Pointnet2MSG(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, num_classes, input_channels=6, use_xyz=True):
        super(Pointnet2MSG, self).__init__()

        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=use_xyz,
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=use_xyz,
            )
        )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
                use_xyz=use_xyz,
            )
        )
        c_out_2 = 256 + 256

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=16,
                radii=[0.4, 0.8],
                nsamples=[16, 32],
                mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
                use_xyz=use_xyz,
            )
        )
        c_out_3 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + input_channels, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))

        self.FC_layer = (
            pt_utils.Seq(128)
            .conv1d(128, bn=True)
            .dropout()
            .conv1d(num_classes, activation=None)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def _first_step(self, l_xyz, l_features):
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        return l_xyz, l_features

    def _second_step(self, l_xyz, l_features):
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features

    def forward(self, pointcloud):
        # type: (Pointnet2MSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        l_xyz, l_features = self._first_step(l_xyz, l_features)
        l_features = self._second_step(l_xyz, l_features)
        return self.FC_layer(l_features[0]).transpose(1, 2).contiguous()


if __name__ == "__main__":
    from torch.autograd import Variable
    import numpy as np
    import torch.optim as optim

    B = 32
    N = 4096
    inputs = torch.randn(B, N, 6).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B * N)).view(B, N).cuda()
    model = Pointnet2MSG(3, input_channels=3)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    print("Testing with xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(50):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        optimizer.step()

    # with use_xyz=False
    inputs = torch.randn(B, N, 6).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B * N)).view(B, N).cuda()
    model = Pointnet2MSG(3, input_channels=3, use_xyz=False)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    print("Testing without xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        optimizer.step()
