import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from models.base_model import MLP, FPModule, UnetBasedModel
from torch_geometric.nn import knn_interpolate
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from .modules import SAModule

class SegmentationModel(UnetBasedModel):
    def __init__(self, opt, num_classes):
        self.down_conv_cls = SAModule
        self.up_conv_cls = FPModule
        self._name = 'POINTNET++_MODEL'
        super(SegmentationModel, self).__init__(opt, num_classes)
        
        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, num_classes)

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