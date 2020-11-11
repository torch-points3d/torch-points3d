import torch.nn as nn

import torchsparse
import torchsparse.nn as spnn
import torchsparse.nn.functional as spf
from torchsparse.sparse_tensor import SparseTensor
from torchsparse.point_tensor import PointTensor
from torchsparse.utils.kernel_region import *
from torchsparse.utils.helpers import *

from torch_points3d.modules.PVCNN.blocks import *
from torch_points3d.modules.PVCNN.utils import *


class PVCNN(nn.Module):
    def __init__(self, option, model_type, dataset, modules):
        super(PVCNN, self).__init__()

        cr = option.cr
        self.vres = option.vres
        self.num_classes = dataset.num_classes
        self.num_features = dataset.feature_dimension

        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        self.stem = nn.Sequential(
            spnn.Conv3d(self.num_features, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1),
                    ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1),
                ),
            ]
        )

        self.up2 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1),
                    ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1),
                ),
            ]
        )

        self.up3 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1),
                    ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1),
                ),
            ]
        )

        self.up4 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1),
                    ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
                ),
            ]
        )

        self.classifier = nn.Sequential(nn.Linear(cs[8], self.num_classes))

        self.point_transforms = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(cs[0], cs[4]), nn.BatchNorm1d(cs[4]), nn.ReLU(True),),
                nn.Sequential(nn.Linear(cs[4], cs[6]), nn.BatchNorm1d(cs[6]), nn.ReLU(True),),
                nn.Sequential(nn.Linear(cs[6], cs[8]), nn.BatchNorm1d(cs[8]), nn.ReLU(True),),
            ]
        )

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

        self.loss_names = ["loss_seg"]

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: SparseTensor z: PointTensor
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, 1.0, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.up1[0](y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)

        self.output = self.classifier(z3.F)
        return self.output
