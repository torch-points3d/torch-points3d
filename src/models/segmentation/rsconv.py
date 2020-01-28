import logging
import queue

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import etw_pytorch_utils as pt_utils

from src.models.base_architectures import UnwrappedUnetBasedModel
from src.modules.RSConv import *
from .base import Segmentation_MP

log = logging.getLogger(__name__)


class RSConvLogicModel(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnwrappedUnetBasedModel
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)
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
            self.FC_layer.conv1d(last_mlp_opt.nn[i], bn=True, bias=False)
        if last_mlp_opt.dropout:
            self.FC_layer.dropout(p=last_mlp_opt.dropout)

        self.FC_layer.conv1d(self._num_classes, activation=None)
        self.loss_names = ["loss_seg"]

        log.info(self)

    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        Sets:
            self.data:
                x -- Features [B, C, N]
                pos -- Features [B, 3, N]
        """
        self.input = Data(x=data.x.transpose(1, 2).contiguous() if data.x is not None else None, pos=data.pos)
        self.labels = torch.flatten(data.y).long()  # [B,N]
        self.batch_idx = torch.arange(0, data.pos.shape[0]).view(-1, 1).repeat(1, data.pos.shape[1]).view(-1)
        if self._use_category:
            self.category = data.category

    def forward(self):
        r"""
            Forward pass of the network
            self.data:
                x -- Features [B, C, N]
                pos -- Features [B, N, 3]
        """
        stack_down = []
        queue_up = queue.Queue()

        data = self.input
        stack_down.append(data)

        for i in range(len(self.down_modules) - 1):
            data = self.down_modules[i](data)
            stack_down.append(data)

        data = self.down_modules[-1](data)
        queue_up.put(data)

        assert len(self.inner_modules) == 2, "For this segmentation model, we except 2 distinct inner"
        data_inner = self.inner_modules[0](data)
        data_inner_2 = self.inner_modules[1](stack_down[3])

        for i in range(len(self.up_modules) - 1):
            data = self.up_modules[i]((queue_up.get(), stack_down.pop()))
            queue_up.put(data)

        last_feature = torch.cat(
            [data.x, data_inner.x.repeat(1, 1, data.x.shape[-1]), data_inner_2.x.repeat(1, 1, data.x.shape[-1])], dim=1
        )

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


class RSConv_MP(Segmentation_MP):
    """ Message passing version of RSConv"""
