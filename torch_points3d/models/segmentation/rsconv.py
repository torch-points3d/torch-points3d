import logging
import queue

import torch
import torch.nn.functional as F

from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.models.base_architectures import UnwrappedUnetBasedModel
from torch_points3d.modules.RSConv import *
from torch_points3d.core.common_modules.dense_modules import Conv1D
from torch_points3d.core.common_modules.base_modules import Seq
from .base import Segmentation_MP

log = logging.getLogger(__name__)


class RSConvLogicModel(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnwrappedUnetBasedModel
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)
        self._num_classes = dataset.num_classes
        self._weight_classes = dataset.weight_classes
        self._use_category = getattr(option, "use_category", False)
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

        self.FC_layer = Seq()
        last_mlp_opt.nn[0] += self._num_categories
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.append(Conv1D(last_mlp_opt.nn[i - 1], last_mlp_opt.nn[i], bn=True, bias=False))
        if last_mlp_opt.dropout:
            self.FC_layer.append(torch.nn.Dropout(p=last_mlp_opt.dropout))

        self.FC_layer.append(Conv1D(last_mlp_opt.nn[-1], self._num_classes, activation=None, bias=True, bn=False))
        self.loss_names = ["loss_seg"]

        self.visual_names = ["data_visual"]

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        Sets:
            self.data:
                x -- Features [B, C, N]
                pos -- Features [B, 3, N]
        """
        data = data.to(device)
        if data.x is not None:
            data.x = data.x.transpose(1, 2).contiguous()
        self.input = data
        if data.y is not None:
            self.labels = torch.flatten(data.y).long()  # [B,N]
        else:
            self.labels = data.y
        self.batch_idx = torch.arange(0, data.pos.shape[0]).view(-1, 1).repeat(1, data.pos.shape[1]).view(-1)
        if self._use_category:
            self.category = data.category

    def forward(self, *args, **kwargs):
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
            cat_one_hot = F.one_hot(self.category, self._num_categories).float().transpose(1, 2)
            last_feature = torch.cat((last_feature, cat_one_hot), dim=1)

        self.output = self.FC_layer(last_feature).transpose(1, 2).contiguous().view((-1, self._num_classes))

        # Compute loss
        if self._weight_classes is not None:
            self._weight_classes = self._weight_classes.to(self.output.device)
        if self.labels is not None:
            self.loss_seg = F.cross_entropy(
                self.output, self.labels, weight=self._weight_classes, ignore_index=IGNORE_LABEL
            )

        self.data_visual = self.input
        self.data_visual.y = torch.reshape(self.labels, data.pos.shape[0:2])
        self.data_visual.pred = torch.max(self.output, -1)[1].reshape(data.pos.shape[0:2])

        return self.output

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_seg.backward()


class RSConv_MP(Segmentation_MP):
    """ Message passing version of RSConv"""
