import logging
import torch.nn.functional as F
import torch

from torch_points3d.modules.MinkowskiEngine import *
from torch_points3d.models.base_architectures import UnwrappedUnetBasedModel
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.applications.minkowski import Minkowski


log = logging.getLogger(__name__)


class Minkowski_Baseline_Model(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        super(Minkowski_Baseline_Model, self).__init__(option)
        self._weight_classes = dataset.weight_classes
        self.model = initialize_minkowski_unet(
            option.model_name, dataset.feature_dimension, dataset.num_classes, **option.get("extra_options", {})
        )
        self.loss_names = ["loss_seg"]

    def set_input(self, data, device):

        self.batch_idx = data.batch.squeeze()
        coords = torch.cat([data.batch.unsqueeze(-1).int(), data.coords.int()], -1)
        self.input = ME.SparseTensor(features=data.x, coordinates=coords, device=device)
        self.labels = data.y.to(device)

    def forward(self, *args, **kwargs):
        self.output = F.log_softmax(self.model(self.input).features, dim=-1)        
        if self._weight_classes is not None:
            self._weight_classes = self._weight_classes.to(self.device)
        if self.labels is not None:
            self.loss_seg = F.nll_loss(self.output, self.labels, ignore_index=IGNORE_LABEL, weight=self._weight_classes)

    def backward(self):
        self.loss_seg.backward()
