import logging
import torch.nn.functional as F
import torch

from src.modules.MinkowskiEngine import *
from src.models.base_architectures import UnwrappedUnetBasedModel
from src.models.base_model import BaseModel


log = logging.getLogger(__name__)


class Minkowski_Baseline_Model(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        super(Minkowski_Baseline_Model, self).__init__(option)

        self.model = initialize_minkowski_unet(
            option.model_name, dataset.feature_dimension, dataset.num_classes, option.D
        )

    def set_input(self, data, device):

        self.batch_idx = data.batch.squeeze()
        coords = torch.cat([data.batch.unsqueeze(-1).int(), data.pos.int()], -1)
        self.input = ME.SparseTensor(data.x, coords=coords).to(device)
        self.labels = data.y.to(device)

    def forward(self):

        self.output = self.model(self.input).feats
        self.loss_seg = F.cross_entropy(self.output, self.labels)

    def backward(self):
        self.loss_seg.backward()


# This model still doesn't fully work yet.
class Minkowski_Model(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnetBasedModel
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)

    def set_input(self, data, device):
        coords = torch.cat([data.batch.unsqueeze(-1).int(), data.pos.int()], -1)
        self.input = ME.SparseTensor(data.x, coords=coords).to(device)
        self.labels = data.y

    def forward(self):

        stack_down = []

        x = self.input
        for i in range(len(self.down_modules) - 1):
            print(x.shape)
            x = self.down_modules[i](x)
            stack_down.append(x)

        x = self.down_modules[-1](x)

        for i in range(len(self.up_modules)):
            x = self.up_modules[i](x, stack_down.pop())

    def backward(self):
        pass
