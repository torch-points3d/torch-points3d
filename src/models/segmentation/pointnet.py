import torch.nn.functional as F
import logging

from src.core.base_conv.base_conv import *
from src.core.common_modules.base_modules import *

from src.modules.PointNet import *
from src.models.base_model import BaseModel
from src.utils.model_building_utils.resolver_utils import flatten_dict

log = logging.getLogger(__name__)


class PointNet(BaseModel):
    def __init__(self, opt, type, dataset, modules_lib):
        super().__init__(opt)

        self.has_fixed_points_transform = (
            dataset.has_fixed_points_transform if hasattr(dataset, "has_fixed_points_transform") else False
        )
        self.pointnet_seg = PointNetSeg(**flatten_dict(opt))
        log.info(self)

    def set_input(self, data):
        self.input = data
        self.labels = data.y

        batch_size = len(data.__slices__["pos"]) - 1
        self.pointnet_seg.set_scatter_pooling(not self.has_fixed_points_transform, batch_size)

    def forward(self):

        x = self.pointnet_seg(self.input.pos, self.input.batch)
        self.output = F.log_softmax(x, dim=-1)

        return self.output

    def backward(self):
        internal_loss = self.get_internal_loss()
        self.loss = F.nll_loss(self.output, self.labels) + (internal_loss if internal_loss.item() != 0 else 0) * 0.001
        self.loss.backward()
