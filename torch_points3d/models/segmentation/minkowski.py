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
        self.model = initialize_minkowski_unet(
            option.model_name, dataset.feature_dimension, dataset.num_classes, **option.get("extra_options", {})
        )
        self.loss_names = ["loss_seg"]

    def set_input(self, data, device):
        self.input = data
        self.batch_idx = data.batch.squeeze()
        self.labels = data.y.to(device)

    def forward(self, epoch=-1, **kwargs):
        coords = torch.cat([self.input.batch.unsqueeze(-1).int(), self.input.coords.int()], -1)
        input = ME.SparseTensor(self.input.x, coords=coords).to(self.device)
        self.output = self.model(input).feats
        self.loss_seg = F.cross_entropy(self.output, self.labels, ignore_index=IGNORE_LABEL)
        with torch.no_grad():
            self._dump_visuals(epoch)

    def backward(self):
        self.loss_seg.backward()
