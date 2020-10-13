import logging
import torch.nn.functional as F
import torch.nn as nn
import torchsparse as TS
import torch

from torch_points3d.modules.SparseConv3D.modules import *
from torch_points3d.models.base_architectures import UnwrappedUnetBasedModel
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL


log = logging.getLogger(__name__)


class UNet(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnetBasedModel
        super().__init__(option, model_type, dataset, modules)
        self.loss_names = ["loss_seg"]
        self.head = nn.Sequential(nn.Linear(option.up_conv.up_conv_nn[-1][-1], dataset.num_classes))

    def set_input(self, data, device):
        self.raw_input = data
        coords = torch.cat([data.coords.int(), data.batch.unsqueeze(-1).int()], -1)
        self.input = SparseTensor(data.x, coords, device)
        self.labels = data.y.to(self.device)

    def forward(self, *args, **kwargs):
        data = self.input
        stack_down = []
        for i in range(len(self.down_modules) - 1):
            data = self.down_modules[i](data)
            stack_down.append(data)

        data = self.down_modules[-1](data)
        stack_down.append(None)
        for i in range(len(self.up_modules)):
            data = self.up_modules[i](data, stack_down.pop())

        feats = data.F
        out = self.head(feats)
        self.output = F.log_softmax(out, dim=-1)
        self.loss_seg = F.nll_loss(self.output, self.labels, ignore_index=IGNORE_LABEL)

    def backward(self):
        self.loss_seg.backward()
