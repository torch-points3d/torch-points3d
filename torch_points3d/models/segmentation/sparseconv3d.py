import logging
import torch.nn.functional as F
import torch.nn as nn
import torchsparse as TS


from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.applications.sparseconv3d import SparseConv3d
import torch_points3d.modules.SparseConv3d as sp3d

from torch_points3d.core.common_modules import FastBatchNorm1d, Seq

log = logging.getLogger(__name__)


class APIModel(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnetBasedModel
        super().__init__(option)
        self._weight_classes = dataset.weight_classes
        self.backbone = SparseConv3d(
            "unet", dataset.feature_dimension, config=option.backbone, backend=option.get("backend", "minkowski")
        )
        self._supports_mixed = sp3d.nn.get_backend() == "torchsparse"
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
