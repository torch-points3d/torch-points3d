
import torch 
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool

from models.core_modules import *
from models.PointNet.modules import *
from models.base_model import BaseModel
from models.model_building_utils.config_utils import flatten_dict

class SegmentationModel(BaseModel):

    def __init__(self, opt, type, dataset, modules_lib):
        super().__init__(opt)

        self.pointnet_seg = PointNetSeg(**flatten_dict(opt))

    def set_input(self, data):
        self.input = data
        self.labels = data.y

    def forward(self):
        
        x = self.pointnet_seg(self.input.pos, self.input.batch)
        self.output = F.log_softmax(x, dim=-1)

        return self.output

    def backward(self):
        self.loss = F.nll_loss(self.output, self.labels)
        self.loss.backward()
    






