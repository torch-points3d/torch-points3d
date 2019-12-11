import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import functional as FPModule
from models.base_model import MLP, FPModule, UnetBasedModel
from models.KPConv import modules

class SegmentationModel(UnetBasedModel):
    def __init__(self, *args, **kwargs):
        super(SegmentationModel, self).__init__(*args, **kwargs)

        nn = args[0].mlp_cls.nn
        self.dropout = args[0].mlp_cls.get('dropout')
        self.lin1 = torch.nn.Linear(nn[0], nn[1])
        self.lin2 = torch.nn.Linear(nn[2], nn[3])
        self.lin3 = torch.nn.Linear(nn[3], args[1])


    def forward(self, data):
        """Standard forward"""
        input = (data.x, data.pos, data.batch)
        x, _, _  = self.model(input)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)   