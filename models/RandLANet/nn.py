
import torch.nn.functional as F
from models.unet_base import UnetBasedModel
from models.core_modules import FPModule
from models.base_model import *

class SegmentationModel(UnetBasedModel):
    def __init__(self, *args, **kwargs):
        super(SegmentationModel, self).__init__(*args, **kwargs)
    
        nn = args[0].mlp_cls.nn
        self.dropout = args[0].mlp_cls.get('dropout')
        self.lin1 = torch.nn.Linear(nn[0], nn[1])
        self.lin2 = torch.nn.Linear(nn[2], nn[3])
        self.lin3 = torch.nn.Linear(nn[3], args[1])

    def set_input(self, data):
        self.input = (data.x if data.x is not None else data.pos, data.pos, data.batch)
        self.labels = data.y

    def forward(self):
        """Standard forward"""
        x, _, _  = self.model(self.input)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        self.output = F.log_softmax(x, dim=-1)   

    def backward(self):
        self.loss_seg = F.nll_loss(self.output, self.labels)
        self.loss_seg.backward()
