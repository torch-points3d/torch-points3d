import logging
import torch.nn.functional as F
from torch_geometric.data import Data
import torch
import os

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

    def _dump_visuals(self, epoch):
        if False:
            if not hasattr(self, "visual_count"):
                self.visual_count = 0
            data_visual = Data(pos=self.input.pos, y=self.input.y, coords=self.input.coords, batch=self.input.batch)
            data_visual.semantic_pred = torch.max(self.output, -1)[1]

            if not os.path.exists("viz"):
                os.mkdir("viz")
            torch.save(data_visual.to("cpu"), "viz/data_e%i_%i.pt" % (epoch, self.visual_count))
            self.visual_count += 1


class APIModel(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnetBasedModel
        super().__init__(option)
        self.model = Minkowski("unet", dataset.feature_dimension, 4, output_nc=dataset.num_classes)
        self.loss_names = ["loss_seg"]

    def set_input(self, data, device):
        self.input = data
        self.labels = data.y.to(self.device)

    def forward(self, epoch=-1, **kwargs):
        self.output = self.model(self.input).x
        self.loss_seg = F.cross_entropy(self.output, self.labels, ignore_index=IGNORE_LABEL)
        with torch.no_grad():
            self._dump_visuals(epoch)

    def backward(self):
        self.loss_seg.backward()

    def _dump_visuals(self, epoch):
        if False:
            if not hasattr(self, "visual_count"):
                self.visual_count = 0
            data_visual = Data(pos=self.input.pos, y=self.input.y, coords=self.input.coords, batch=self.input.batch)
            data_visual.semantic_pred = torch.max(self.output, -1)[1]

            if not os.path.exists("viz"):
                os.mkdir("viz")
            torch.save(data_visual.to("cpu"), "viz/data_e%i_%i.pt" % (epoch, self.visual_count))
            self.visual_count += 1
