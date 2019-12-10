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
        dim_out = opt.convs.final_up_conv_nn[-1]
        self.lin1 = torch.nn.Linear(dim_out, dim_out)
        self.lin2 = torch.nn.Linear(dim_out, dim_out)
        self.lin3 = torch.nn.Linear(dim_out, num_classes)

        print(self)
        
    def forward(self, data):
        """Standard forward"""
        input = (data.x, data.pos, data.batch)
        data_out = self.model(input)
        data = (*data_out, *input)
        x, _, _ = self.upconv(data)
        
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)   