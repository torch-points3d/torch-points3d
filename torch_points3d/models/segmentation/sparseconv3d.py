import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchsparse as TS


from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.applications.sparseconv3d import SparseConv3d

from torch_points3d.core.common_modules import FastBatchNorm1d, Seq
from torch_points3d.models.registration.ms_spconv3d import UnetMSparseConv3d

log = logging.getLogger(__name__)


class APIModel(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnetBasedModel
        super().__init__(option)
        self._weight_classes = dataset.weight_classes
        self.backbone = SparseConv3d(
            "unet", dataset.feature_dimension, config=option.backbone, backend=option.get("backend", "minkowski")
        )
        self.head = nn.Sequential(nn.Linear(self.backbone.output_nc, dataset.num_classes))
        self.loss_names = ["loss_seg"]

    def set_input(self, data, device):
        self.batch_idx = data.batch.squeeze()
        self.input = data
        if data.y is not None:
            self.labels = data.y.to(self.device)
        else:
            self.labels = None

    def forward(self, *args, **kwargs):
        features = self.backbone(self.input).x
        logits = self.head(features)
        self.output = F.log_softmax(logits, dim=-1)     
        if self._weight_classes is not None:
            self._weight_classes = self._weight_classes.to(self.device)
        if self.labels is not None:
            self.loss_seg = F.nll_loss(self.output, self.labels, ignore_index=IGNORE_LABEL, weight=self._weight_classes)

    def backward(self):
        self.loss_seg.backward()


class MS_SparseConvModel(APIModel):
    def __init__(self, option, model_type, dataset, modules):
        BaseModel.__init__(self, option)
        option_unet = option.option_unet
        self.normalize_feature = option.normalize_feature
        self.grid_size = option_unet.grid_size
        self.unet = UnetMSparseConv3d(
            option_unet.backbone,
            input_nc=option_unet.input_nc,
            pointnet_nn=option_unet.pointnet_nn,
            post_mlp_nn=option_unet.post_mlp_nn,
            pre_mlp_nn=option_unet.pre_mlp_nn,
            add_pos=option_unet.add_pos,
            add_pre_x=option_unet.add_pre_x,
            aggr=option_unet.aggr,
            backend=option.backend,
        )
        if option.mlp_cls is not None:
            last_mlp_opt = option.mlp_cls

            self.FC_layer = Seq()
            for i in range(1, len(last_mlp_opt.nn)):
                self.FC_layer.append(
                    nn.Sequential(
                        *[
                            nn.Linear(last_mlp_opt.nn[i - 1], last_mlp_opt.nn[i], bias=False),
                            FastBatchNorm1d(last_mlp_opt.nn[i], momentum=last_mlp_opt.bn_momentum),
                            nn.LeakyReLU(0.2),
                        ]
                    )
                )
            if last_mlp_opt.dropout:
                self.FC_layer.append(Dropout(p=last_mlp_opt.dropout))
        else:
            self.FC_layer = torch.nn.Identity()
        self.head = nn.Sequential(nn.Linear(option.output_nc, dataset.num_classes))
        self.loss_names = ["loss_seg"]

    def apply_nn(self, input):

        outputs = []
        for i in range(len(self.grid_size)):
            self.unet.set_grid_size(self.grid_size[i])
            out = self.unet(input.clone())
            out.x = out.x / (torch.norm(out.x, p=2, dim=1, keepdim=True) + 1e-20)
            outputs.append(out)
        x = torch.cat([o.x for o in outputs], 1)
        out_feat = self.FC_layer(x)
        if self.normalize_feature:
            out_feat = out_feat / (torch.norm(out_feat, p=2, dim=1, keepdim=True) + 1e-20)
        out_feat = self.head(out_feat)
        return out_feat, outputs

    def forward(self, *args, **kwargs):
        logits, _ = self.apply_nn(self.input)
        self.output = F.log_softmax(logits, dim=-1)
        if self.labels is not None:
            self.loss_seg = F.nll_loss(self.output, self.labels, ignore_index=IGNORE_LABEL)

    def backward(self):
        self.loss_seg.backward()
