import logging
import torch
import torch.nn.functional as F
from typing import Any
from torch_points3d.datasets.multiscale_data import MultiScaleBatch
from torch_points3d.models.base_architectures import BackboneBasedModel
from torch_points3d.models.base_model import BaseModel
from torch_points3d.core.common_modules.dense_modules import Conv1D
from torch_points3d.core.common_modules.base_modules import Seq

log = logging.getLogger(__name__)


def create_batch_siamese(pair, batch):
    """
    create a batch with siamese input
    """
    return 2 * batch + pair


class PatchSiamese(BackboneBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        """
        Initialize this model class
        Parameters:
            opt -- training/test options
        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """

        BackboneBasedModel.__init__(self, option, model_type, dataset, modules)
        self.set_last_mlp(option.mlp_cls)
        self.loss_names = ["loss_reg"]

    def set_input(self, data, device):
        data = data.to(device)
        self.input = data
        # TODO multiscale data pre_computed...
        if isinstance(data, MultiScaleBatch):
            self.pre_computed = data.multiscale
            del data.multiscale
        else:
            self.pre_computed = None
        # batch siamese
        self.batch_idx = create_batch_siamese(data.pair, data.batch)

    def set_last_mlp(self, last_mlp_opt):
        self.FC_layer = Seq()
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.append(Conv1D(last_mlp_opt.nn[i - 1], last_mlp_opt.nn[i], bn=True, bias=False))

    def set_loss(self):
        raise NotImplementedError("Choose a loss for the metric learning")

    def forward(self, *args, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""

        data = self.input
        for i in range(len(self.down_modules)):
            data = self.down_modules[i](data, precomputed=self.pre_computed)

        x = F.relu(self.lin1(data.x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        self.output = self.lin2(x)

        self.loss_reg = self.loss_module(self.output) + self.get_internal_loss()
        return self.output

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_reg.backward()  # calculate gradients of network G w.r.t. loss_G


class FragmentBaseModel(BaseModel):
    def __init__(self, option):
        BaseModel.__init__(self, option)

    def set_input(self, data, device):
        raise NotImplementedError("need to define set_input")

    def compute_loss_match(self):
        if hasattr(self, "xyz"):
            xyz = self.xyz
            xyz_target = self.xyz_target
        else:
            xyz = self.input.pos
            xyz_target = self.input_target.pos
        loss_reg = self.metric_loss_module(self.output, self.output_target, self.match[:, :2], xyz, xyz_target)
        return loss_reg

    def compute_loss_label(self):
        """
        compute the loss separating the miner and the loss
        each point correspond to a labels
        """
        output = torch.cat([self.output[self.match[:, 0]], self.output_target[self.match[:, 1]]], 0)
        rang = torch.arange(0, len(self.match), dtype=torch.long, device=self.match.device)
        labels = torch.cat([rang, rang], 0)
        hard_pairs = None
        if self.miner_module is not None:
            hard_pairs = self.miner_module(output, labels)
        # loss
        loss_reg = self.metric_loss_module(output, labels, hard_pairs)
        return loss_reg

    def compute_loss(self):
        if self.mode == "match":
            self.loss = self.compute_loss_match()
        elif self.mode == "label":
            self.loss = self.compute_loss_label()
        else:
            raise NotImplementedError("The mode for the loss is incorrect")

    def apply_nn(self, input):
        raise NotImplementedError("Model still not defined")

    def forward(self, *args, **kwargs):
        self.output = self.apply_nn(self.input)
        if self.match is None:
            return self.output

        self.output_target = self.apply_nn(self.input_target)
        self.compute_loss()

        return self.output

    def backward(self):
        if hasattr(self, "loss"):
            self.loss.backward()

    def get_output(self):
        if self.match is not None:
            return self.output, self.output_target
        else:
            return self.output, None

    def get_batch(self):
        raise NotImplementedError("Need to define get_batch")

    def get_input(self):
        raise NotImplementedError("Need to define get_input")
