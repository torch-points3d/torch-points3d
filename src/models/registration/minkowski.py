import logging
import torch.nn.functional as F
import torch

from src.modules.MinkowskiEngine import *
from src.models.base_architectures import UnwrappedUnetBasedModel
from src.models.base_model import BaseModel


log = logging.getLogger(__name__)


class Minkowski_Baseline_Model_Fragment(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        super(Minkowski_Baseline_Model_Fragment, self).__init__(option)
        self.model = initialize_minkowski_unet(
            option.model_name, dataset.feature_dimension, dataset.num_classes, option.D
        )
        self.loss_names = ["loss_reg"]

    def set_input(self, data, device):

        self.batch_idx = data.batch.squeeze()
        coords = torch.cat([data.batch.unsqueeze(-1).int(), data.pos.int()], -1)
        self.input = ME.SparseTensor(data.x, coords=coords).to(device)

        if hasattr(data, "pos_target"):
            coords_target = torch.cat([data.batch_target.unsqueeze(-1).int(), data.pos_target.int()], -1)
            self.input_target = ME.SparseTensor(data.x_target, coords=coords_target).to(device)
            self.labels = data.y.to(device)
        else:
            self.labels = None

    def apply_nn(self, input):
        output = self.model(self.input).feats
        return F.normalize(output, dim=-1)

    def forward(self):
        self.output = self.apply_nn(self.input)
        if self.labels is None:
            return self.output
        else:
            self.output_target = self.apply_nn(self.input_target)
            self.compute_loss()
            return

    def backward(self):
        self.loss_seg.backward()


class MinkowskiFragment(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnetBasedModel
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)
        self.loss_names = ["loss_reg"]

    def set_input(self, data, device):
        coords = torch.cat([data.batch.unsqueeze(-1).int(), data.pos.int()], -1)
        self.input = ME.SparseTensor(data.x, coords=coords).to(device)

        if hasattr(data, "pos_target"):
            coords_target = torch.cat([data.batch_target.unsqueeze(-1).int(), data.pos_target.int()], -1)
            self.input_target = ME.SparseTensor(data.x_target, coords=coords_target).to(device)
            self.labels = data.y.to(device)
        else:
            self.labels = None

    def apply_nn(self, input):
        x = input
        stack_down = []
        for i in range(len(self.down_modules) - 1):
            print(x.shape)
            x = self.down_modules[i](x)
            stack_down.append(x)

        x = self.down_modules[-1](x)

        for i in range(len(self.up_modules)):
            x = self.up_modules[i](x, stack_down.pop())
            print(x.shape)
        return F.normalize(x.feats, p=2, dim=-1)

    def compute_loss(self):
        self.loss = 0

    def forward(self):

        self.output = self.apply_nn(self.input)
        if self.labels is None:
            return self.output
        else:
            self.output_target = self.apply_nn(self.input_target)
            self.compute_loss()
            return

    def backward(self):
        self.loss.backward()
