import torch.nn.functional as F
import logging

from src.core.base_conv.base_conv import *
from src.core.common_modules.base_modules import *
from src.utils.config import ConvolutionFormatFactory
from src.modules.PointNet import *
from src.models.base_model import BaseModel
from src.utils.model_building_utils.resolver_utils import flatten_dict

log = logging.getLogger(__name__)


class PointNet(BaseModel):
    def __init__(self, opt, type, dataset, modules_lib):
        super().__init__(opt)
        self.pointnet_seg = PointNetSeg(**flatten_dict(opt))
        self._is_dense = ConvolutionFormatFactory.check_is_dense_format(self.conv_type)

    def set_input(self, data, device):
        data = data.to(device)
        self.input = data
        self.labels = data.y

        self.pointnet_seg.set_scatter_pooling(not self._is_dense)

    def forward(self):

        x = self.pointnet_seg(self.input.pos, self.input.batch)
        self.output = F.log_softmax(x, dim=-1)
        internal_loss = self.get_internal_loss()
        if self.labels is not None:
            self.loss = (
                F.nll_loss(self.output, self.labels) + (internal_loss if internal_loss.item() != 0 else 0) * 0.001
            )
        return self.output

    def backward(self):
        self.loss.backward()
