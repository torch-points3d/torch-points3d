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
        self.mlp_cls = MLP(opt.mlp_cls + [num_classes])
        
    def forward(self, data):
        input = (data.x, data.pos, data.batch)
        output = self.model(input)
        return F.log_softmax(self.mlp_cls(output[0]), dim=-1)