import torch

import torch.nn.functional as F
from torch_geometric.data import Data
import logging

from torch_points3d.modules.pointnet2 import *
from torch_points3d.core.base_conv.dense import DenseFPModule
from torch_points3d.models.base_architectures import BackboneBasedModel
from torch_points3d.core.common_modules.dense_modules import Conv1D
from torch_points3d.core.common_modules.base_modules import Seq
from torch_points3d.datasets.segmentation import IGNORE_LABEL
import torch_points3d.core.base_conv.dense as dense_modules

log = logging.getLogger(__name__)


class PointNet2_D(BackboneBasedModel):
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
        BackboneBasedModel.__init__(self, option, model_type, dataset, modules)
        self._num_classes = dataset.num_classes
        self._weight_classes = dataset.weight_classes

        innermost_option = option.innermost
        innermost_cls = getattr(dense_modules, innermost_option.module_name)
        self.innermost_module = innermost_cls(**innermost_option)

        # Last MLP
        last_mlp_opt = option.mlp_cls

        self.FC_layer = Seq()
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.append(Conv1D(last_mlp_opt.nn[i - 1], last_mlp_opt.nn[i], bn=True, bias=False))
        if last_mlp_opt.dropout:
            self.FC_layer.append(torch.nn.Dropout(p=last_mlp_opt.dropout))

        self.FC_layer.append(Conv1D(last_mlp_opt.nn[-1], self._num_classes, activation=None, bias=True, bn=False))
        self.loss_names = ["loss_cls"]

        # self.visual_names = ["data_visual"]

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        Sets:
            self.input:
                x -- Features [B, C, N]
                pos -- Points [B, N, 3]
        """
        assert len(data.pos.shape) == 3
        data = data.to(device)
        if data.x is not None:
            x = data.x.transpose(1, 2).contiguous()
        else:
            x = None
        self.input = Data(x=x, pos=data.pos)
        if data.y is not None:
            self.labels = torch.flatten(data.y).long()  # [B * N]
        else:
            self.labels = None
        self.batch_idx = torch.arange(0, data.pos.shape[0]).view(-1, 1).repeat(1, data.pos.shape[1]).view(-1)

    def forward(self, *args, **kwargs):
        r"""
            Forward pass of the network
            self.input:
                x -- Features [B, C, N]
                pos -- Points [B, N, 3]
        """
        data = self.input
        labels = data.y
        for i in range(len(self.down_modules)):
            data = self.down_modules[i](data)
        data = self.innermost_module(data)

        last_feature = data.x

        self.output = self.FC_layer(last_feature).transpose(1, 2).contiguous().view((-1, self._num_classes))

        if self._weight_classes is not None:
            self._weight_classes = self._weight_classes.to(self.output.device)
        if self.labels is not None:
            self.loss_cls = F.cross_entropy(
                self.output, self.labels, weight=self._weight_classes, ignore_index=IGNORE_LABEL
            )
            
        return self.output

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_cls.backward()
