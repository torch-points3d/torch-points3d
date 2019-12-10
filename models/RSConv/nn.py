
import torch.nn.functional as F
from models.base_model import UnetBasedModel, FPModule
from .modules import RSConv

class SegmentationModel(UnetBasedModel):
    def __init__(self, opt, num_classes):
        self.down_conv_cls = RSConv
        self.up_conv_cls = FPModule
        self._name = "RS_CONV_MODEL"
        super(SegmentationModel, self).__init__(opt, num_classes)
    
    def forward(self, data):
        inp = (data.x, data.pos, data.batch)
        output = self.model(inp)
        return F.log_softmax(output, dim=-1)
