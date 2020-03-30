import logging
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
        self.loss_module, self.miner_module = self.get_loss_and_miner(
            getattr(option, "loss", None), getattr(option, "miner", None)
        )
        self.num_pos_pairs = option.num_pos_pairs
        self.loss_names = ["loss_reg", "loss", "internal"]

    def set_input(self, data, device):

        self.batch_idx = data.batch.squeeze()
        coords = torch.cat([data.batch.unsqueeze(-1).int(), data.pos.int()], -1)
        self.input = ME.SparseTensor(data.x, coords=coords).to(device)

        if hasattr(data, "pos_target"):
            coords_target = torch.cat([data.batch_target.unsqueeze(-1).int(), data.pos_target.int()], -1)
            self.input_target = ME.SparseTensor(data.x_target, coords=coords_target).to(device)
            num_pos_pairs = len(data.y)
            if self.num_pos_pairs < len(data.y):
                num_pos_pairs = self.num_pos_pairs
            rand_ind = self.randperm(len(data.y))[:num_pos_pairs]
            self.ind = data.y[rand_ind:, 0].to(device)
            self.ind_target = data.y[rand_ind:, 1].to(device)
            rang = torch.range(0, self.num_pos_pairs)
            self.labels = torch.cat([rang, rang], 0).to(device)
        else:
            self.labels = None

    def apply_nn(self, input):
        output = self.model(self.input)
        return ME.SparseTensor(
            output.F / torch.norm(output.F, p=2, dim=1, keepdim=True),
            coords_key=output.coords_key,
            coords_manager=output.coords_man,
        )

    def compute_loss(self):

        # miner
        hard_pairs = None
        if self.miner_module is not None:
            hard_pairs = self.miner_module(self.output, self.labels)

        # loss
        self.loss_reg = self.loss_module(feat, self.labels, hard_pairs)
        self.internal = self.get_internal_loss()
        self.loss = self.loss_reg + self.internal

    def forward(self):
        self.output = self.apply_nn(self.input)
        if self.labels is None:
            return self.output
        else:
            self.apply_nn(self.input_target)
            self.output = torch.cat([self.output[self.ind], self.output_target[self.ind_target]], 0)
            self.compute_loss()
            return

    def backward(self):
        self.loss.backward()


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
            num_pos_pairs = len(data.y)
            if self.num_pos_pairs < len(data.y):
                num_pos_pairs = self.num_pos_pairs
            rand_ind = self.randperm(len(data.y))[:num_pos_pairs]
            self.ind = data.y[rand_ind:, 0].to(device)
            self.ind_target = data.y[rand_ind:, 1].to(device)
            rang = torch.range(0, self.num_pos_pairs)
            self.labels = torch.cat([rang, rang], 0).to(device)
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

        return ME.SparseTensor(
            x.F / torch.norm(x.F, p=2, dim=1, keepdim=True), coords_key=x.coords_key, coords_manager=x.coords_man
        )

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
