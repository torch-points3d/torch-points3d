from typing import Any
import torch

from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import knn_interpolate
from torch_geometric.nn import radius, global_max_pool
from torch_geometric.data import Data
import etw_pytorch_utils as pt_utils

from .modules import *
from models.dense_modules import DenseFPModule
from models.unet_base import UnetBasedModel, BaseModel


class SegmentationModel(BaseModel):
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

    def __init__(self, option, model_type, dataset, modules):
        super(SegmentationModel, self).__init__(option)
        use_xyz = True
        self.loss_names = ["loss_seg"]
        self._weight_classes = dataset.weight_classes
        self._num_classes = dataset.num_classes

        # Downconv
        downconv_opt = option.down_conv
        self.SA_modules = nn.ModuleList()
        for i in range(len(downconv_opt.down_conv_nn)):
            self.SA_modules.append(
                PointNetMSGDown(
                    npoint=downconv_opt.npoint[i],
                    radii=downconv_opt.radii[i],
                    nsamples=downconv_opt.nsamples[i],
                    mlps=downconv_opt.down_conv_nn[i],
                    use_xyz=use_xyz,
                )
            )

        # Up conv
        up_conv_opt = option.up_conv
        self.FP_modules = nn.ModuleList()
        for i in range(len(up_conv_opt.up_conv_nn)):
            self.FP_modules.append(DenseFPModule(up_conv_nn=up_conv_opt.up_conv_nn[i]))

        # Last MLP
        last_mlp_opt = option.mlp_cls
        self.FC_layer = pt_utils.Seq(last_mlp_opt.nn[0])
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.conv1d(last_mlp_opt.nn[i], bn=True)
        if last_mlp_opt.dropout:
            self.FC_layer.dropout(p=last_mlp_opt.dropout)

        self.FC_layer.conv1d(self._num_classes, activation=None)
        print(self)

    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        Sets:
            self.data:
                x -- Features [B, C, N]
                pos -- Features [B, 3, N]
        """
        self.data = Data(x=data.x.transpose(1, 2).contiguous(), pos=data.pos)
        self.labels = torch.flatten(data.y).long()  # [B,N]

    def forward(self):
        r"""
            Forward pass of the network
            self.data:
                x -- Features [B, C, N]
                pos -- Features [B, 3, N]
        """
        datas = [self.data]
        for i in range(len(self.SA_modules)):
            datas.append(self.SA_modules[i](datas[i]))

        for i in range(len(self.FP_modules)):
            datas[-i - 2].x = self.FP_modules[i]((datas[-i - 1], datas[-i - 2]))

        last_feature = datas[0].x
        self.output = self.FC_layer(last_feature).transpose(1, 2).contiguous().view((-1, self._num_classes))
        return self.output

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        if self._weight_classes is not None:
            self._weight_classes = self._weight_classes.to(self.output.device)
        self.loss_seg = F.cross_entropy(self.output, self.labels, weight=self._weight_classes)
        self.loss_seg.backward()
