from typing import Any
import torch

from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import knn_interpolate
from torch_geometric.nn import radius, global_max_pool
from torch_geometric.data import Data
import etw_pytorch_utils as pt_utils
import logging

from .modules import *
from models.dense_modules import DenseFPModule
from models.unet_base import UnetBasedModel

log = logging.getLogger(__name__)


class SegmentationModel(UnetBasedModel):
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
        # call the initialization method of UnetBasedModel
        UnetBasedModel.__init__(self, option, model_type, dataset, modules)
        self._num_classes = dataset.num_classes
        self._weight_classes = dataset.weight_classes
        self._use_category = option.use_category
        if self._use_category:
            if not dataset.class_to_segments:
                raise ValueError(
                    "The dataset needs to specify a class_to_segments property when using category information for segmentation"
                )
            self._num_categories = len(dataset.class_to_segments.keys())
            log.info("Using category information for the predictions with %i categories", self._num_categories)
        else:
            self._num_categories = 0

        # Last MLP
        last_mlp_opt = option.mlp_cls

        self.FC_layer = pt_utils.Seq(last_mlp_opt.nn[0] + self._num_categories)
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.conv1d(last_mlp_opt.nn[i], bn=True)
        if last_mlp_opt.dropout:
            self.FC_layer.dropout(p=last_mlp_opt.dropout)

        self.FC_layer.conv1d(self._num_classes, activation=None)
        self.loss_names = ["loss_seg"]

    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        Sets:
            self.input:
                x -- Features [B, C, N]
                pos -- Points [B, N, 3]
        """
        assert len(data.pos.shape) == 3
        self.input = Data(x=data.x.transpose(1, 2).contiguous(), pos=data.pos)
        self.labels = torch.flatten(data.y).long()  # [B * N]
        self.batch_idx = torch.arange(0, data.pos.shape[0]).view(-1, 1).repeat(1, data.pos.shape[1]).view(-1)
        if self._use_category:
            self.category = data.category

    def forward(self):
        r"""
            Forward pass of the network
            self.input:
                x -- Features [B, C, N]
                pos -- Points [B, N, 3]
        """
        data = self.model(self.input)
        last_feature = data.x
        if self._use_category:
            num_points = data.pos.shape[1]
            cat_one_hot = (
                torch.zeros((data.pos.shape[0], self._num_categories, num_points)).float().to(self.category.device)
            )
            cat_one_hot.scatter_(1, self.category.repeat(1, num_points).unsqueeze(1), 1)
            last_feature = torch.cat((last_feature, cat_one_hot), dim=1)

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
