import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import functional as FPModule
from models.base_model import MLP, FPModule, UnetBasedModel
from .modules import SAModule

class SegmentationModel(UnetBasedModel):
    def __init__(self, opt, num_classes):
        self.down_conv_cls = SAModule
        self.up_conv_cls = FPModule
        self._name = 'POINTNET++_MODEL'
        super(SegmentationModel, self).__init__(opt, num_classes)
        self.mlp_cls = MLP(opt.mlp_cls + [num_classes], p_dropout=0.1)
        
    def forward(self, data):
        """Standard forward"""
        input = (data.x, data.pos, data.batch)
        data_out = self.model(input)
        data = (*data_out, *input)
        x, _, _ = self.upconv(data)
        return F.log_softmax(self.mlp_cls(x), dim=-1)