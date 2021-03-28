import logging
import torch.nn.functional as F
import torch

from torch_points3d.modules.PVCNN import pvcnn
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL

from torchsparse import SparseTensor

log = logging.getLogger(__name__)


class PVCNN(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        super(PVCNN, self).__init__(option)
        self._weight_classes = dataset.weight_classes

        self.model = pvcnn.PVCNN(option, model_type, dataset, modules)

        self.loss_names = ["loss_seg"]

    def set_input(self, data, device):
        if data.batch.dim() == 1:
            data.batch = data.batch.unsqueeze(-1)
        coords = torch.cat([data.pos, data.batch], -1)
        self.batch_idx = data.batch.squeeze()
        self.input = SparseTensor(data.x, coords).to(self.device)
        self.labels = data.y.to(self.device)

    def forward(self, *args, **kwargs):
        self.output = self.model(self.input)            
        if self._weight_classes is not None:
            self._weight_classes = self._weight_classes.to(self.device)
        if self.labels is not None:
            self.loss_seg = F.cross_entropy(
                self.output, self.labels, weight=self._weight_classes, ignore_index=IGNORE_LABEL
            )

    def backward(self):
        self.loss_seg.backward()
