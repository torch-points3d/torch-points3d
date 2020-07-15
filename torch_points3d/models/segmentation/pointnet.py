import torch.nn.functional as F
import logging

from torch_points3d.core.base_conv.base_conv import *
from torch_points3d.core.common_modules.base_modules import *
from torch_points3d.utils.config import ConvolutionFormatFactory
from torch_points3d.modules.PointNet import *
from torch_points3d.models.base_model import BaseModel
from torch_points3d.utils.model_building_utils.resolver_utils import flatten_dict
from torch_points3d.datasets.segmentation import IGNORE_LABEL

log = logging.getLogger(__name__)


class PointNet(BaseModel):
    def __init__(self, opt, type, dataset, modules_lib):
        super().__init__(opt)
        self.pointnet_seg = PointNetSeg(**flatten_dict(opt))
        self._is_dense = ConvolutionFormatFactory.check_is_dense_format(self.conv_type)

        self.visual_names = ["data_visual"]

    def set_input(self, data, device):
        data = data.to(device)
        self.input = data
        self.labels = data.y

        self.pointnet_seg.set_scatter_pooling(not self._is_dense)

    def forward(self, *args, **kwargs):
        x = self.pointnet_seg(self.input.pos, self.input.batch)
        self.output = F.log_softmax(x, dim=-1)
        internal_loss = self.get_internal_loss()
        if self.labels is not None:
            self.loss = (
                F.nll_loss(self.output, self.labels, ignore_index=IGNORE_LABEL)
                + (internal_loss if internal_loss.item() != 0 else 0) * 0.001
            )

        self.data_visual = self.input
        self.data_visual.pred = torch.max(self.output, -1)[1]
        return self.output

    def backward(self):
        self.loss.backward()


class SegPointNetModel(BaseModel):
    def __init__(self, opt, type, dataset, modules_lib):
        super().__init__(opt)
        self.pointnet_seg = MiniPointNet(
            opt.pointnet.local_nn,
            opt.pointnet.global_nn,
            aggr=opt.pointnet.aggr,
            return_local_out=opt.pointnet.return_local_out,
        )
        self.seg_nn = MLP(opt.seg_nn)

    def set_input(self, data, device):
        data = data.to(device)
        self.pos = data.pos
        self.labels = data.y
        if not hasattr(data, "batch"):
            self.batch_idx = torch.zeros(self.labels.shape[0]).long()
        else:
            self.batch_idx = data.batch

    def get_local_feat(self):
        return self.pointnet_seg.local_nn(self.pos)

    def forward(self, *args, **kwargs):
        x = self.pointnet_seg.forward_embedding(self.pos, self.batch_idx)
        x = self.seg_nn(x)
        self.output = F.log_softmax(x, dim=-1)
        self.loss = F.nll_loss(self.output, self.labels)
        return self.output

    def backward(self):
        self.loss.backward()
