import torch.nn.functional as F
import logging
from omegaconf import OmegaConf
from torch_points3d.core.base_conv.base_conv import *
from torch_points3d.core.common_modules.base_modules import *
from torch_points3d.utils.config import ConvolutionFormatFactory
from torch_points3d.modules.PointNet import *
from torch_points3d.models.base_model import BaseModel
from torch_points3d.utils.model_building_utils.resolver_utils import flatten_dict
from torch_points3d.datasets.segmentation import IGNORE_LABEL

log = logging.getLogger(__name__)


class PointNet(BaseModel):
    def __init__(self, opt, model_type=None, dataset=None, modules=None):
        super().__init__(opt)

        self._opt = OmegaConf.to_container(opt)
        self._is_dense = ConvolutionFormatFactory.check_is_dense_format(self.conv_type)

        self._build_model()

        self.loss_names = ["loss_seg", "loss_internal"]
        
        self.visual_names = ["data_visual"]

    def set_input(self, data, device):
        data = data.to(device)
        self.input = data
        if data.x is not None:
            self.input_features = torch.cat([data.pos, data.x], axis=-1)
        else:
            self.input_features = data.pos
        if data.y is not None:
            self.labels = data.y
        else:
            self.labels = None
        if not hasattr(data, "batch"):
            self.batch_idx = torch.zeros(self.labels.shape[0]).long()
        else:
            self.batch_idx = data.batch
        self.pointnet_seg.set_scatter_pooling(not self._is_dense)

    def _build_model(self):
        if not hasattr(self, "pointnet_seg"):
            self.pointnet_seg = PointNetSeg(**flatten_dict(self._opt))

    def forward(self, *args, **kwargs):
        x = self.pointnet_seg(self.input_features, self.input.batch)
        self.output = x
        
        internal_loss = self.get_internal_loss()
        
        if self.labels is not None:
            self.loss_seg = F.cross_entropy(
                self.output, self.labels, ignore_index=IGNORE_LABEL
            )
            self.loss_internal = (internal_loss if internal_loss.item() != 0 else 0) * 0.001
            self.loss = self.loss_seg + self.loss_internal

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
