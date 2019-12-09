import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import functional as FPModule
from models.base_model import MLP, FPModule, UnetBasedModel
from .modules import KPConv

class SegmentationModel(UnetBasedModel):
    def __init__(self, opt, num_classes):
        self.down_conv_cls = KPConv
        self.up_conv_cls = FPModule
        self._name = 'KP_CONV_MODEL'
        super(SegmentationModel, self).__init__(opt, num_classes)
        
    def forward(self, data):
        input = (data.x, data.pos, data.batch)
        output = self.model(input)
        return F.log_softmax(output, dim=-1)