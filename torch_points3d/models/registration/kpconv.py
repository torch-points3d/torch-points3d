from typing import Any
import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin

from torch.nn import Sequential as Seq
from torch.nn import Identity
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool


import logging

from torch_points3d.core.losses import *
from torch_points3d.core.common_modules import MLP
from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.core.base_conv.partial_dense import *

from torch_points3d.models.registration.base import FragmentBaseModel
from torch_points3d.models.base_architectures import BackboneBasedModel
from torch_points3d.models.registration.base import create_batch_siamese
from torch_points3d.models.base_model import BaseModel
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel

from torch_points3d.modules.KPConv import *

from torch_points3d.datasets.registration.pair import PairMultiScaleBatch

log = logging.getLogger(__name__)


class PatchKPConv(BackboneBasedModel):
    r"""
    siamese neural network using Kernel Point
    Convolution to learn descriptors on patch(for registration).
    """

    def __init__(self, option, model_type, dataset, modules):

        BackboneBasedModel.__init__(self, option, model_type, dataset, modules)
        self.set_last_mlp(option.mlp_cls)
        self.loss_names = ["loss_reg", "loss", "internal"]

    def set_last_mlp(self, last_mlp_opt):

        if len(last_mlp_opt.nn) > 2:

            self.FC_layer = MLP(last_mlp_opt.nn[: len(last_mlp_opt.nn) - 1])
            self.FC_layer.add_module("last", Lin(last_mlp_opt.nn[-2], last_mlp_opt.nn[-1]))
        elif len(last_mlp_opt.nn) == 2:
            self.FC_layer = Seq(Lin(last_mlp_opt.nn[-2], last_mlp_opt.nn[-1]))
        else:
            self.FC_layer = torch.nn.Identity()

    def set_input(self, data, device):
        data = data.to(device)
        data.x = add_ones(data.pos, data.x, True)
        self.batch_idx = data.batch
        if isinstance(data, PairMultiScaleBatch):
            self.pre_computed = data.multiscale
        else:
            self.pre_computed = None
        if getattr(data, "pos_target", None) is not None:
            data.x_target = add_ones(data.pos_target, data.x_target, True)
            if isinstance(data, PairMultiScaleBatch):
                self.pre_computed_target = data.multiscale_target
                del data.multiscale_target
            else:
                self.pre_computed_target = None

            self.input, self.input_target = data.to_data()
            self.batch_idx_target = data.batch_target
            rang = torch.arange(0, data.batch_idx[-1])
            rang_target = torch.arange(0, data.batch_idx_target[-1])
            assert len(rang) == len(rang_target)
            self.labels = torch.cat([rang, rang_target], 0).to(device)
        else:
            self.input = data
            self.labels = None

    def apply_nn(self, input, pre_computed, batch):
        data = input
        for i in range(len(self.down_modules)):
            data = self.down_modules[i](data, precomputed=pre_computed)

        last_feature = global_mean_pool(data.x, batch)
        output = self.FC_layer(last_feature)
        return F.normalize(output, p=2, dim=1)

    def forward(self, *args, **kwargs) -> Any:

        self.output = self.apply_nn(self.input, self.pre_computed, self.batch_idx)
        if self.labels is None:
            return self.output
        else:
            output_target = self.apply_nn(self.input_target, self.pre_computed_target, self.batch_idx_target)
            self.output = torch.cat([self.output, output_target], 0)
            self.compute_loss()
            return self.output

    def compute_loss(self):
        self.loss = 0

        # Get regularization on weights
        if self.lambda_reg:
            self.loss_reg = self.get_regularization_loss(regularizer_type="l2", lambda_reg=self.lambda_reg)
            self.loss += self.loss_reg

        # Collect internal losses and set them with self and them to self for later tracking
        if self.lambda_internal_losses:
            self.loss += self.collect_internal_losses(lambda_weight=self.lambda_internal_losses)

        hard_pairs = None
        if self.miner_module is not None:
            hard_pairs = self.miner_module(self.output, self.labels)
        self.loss_reg = self.metric_loss_module(self.output, self.labels, hard_pairs)

        self.loss += self.loss_reg

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        if hasattr(self, "loss"):
            self.loss.backward()  # calculate gradients of network G w.r.t. loss_G


class FragmentKPConv(FragmentBaseModel, UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # Extract parameters from the dataset
        # Assemble encoder / decoder
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)

        # Build final MLP
        last_mlp_opt = option.mlp_cls

        self.out_channels = option.out_channels
        in_feat = last_mlp_opt.nn[0]
        self.FC_layer = Seq()
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.add_module(
                str(i),
                Seq(
                    *[
                        Lin(in_feat, last_mlp_opt.nn[i], bias=False),
                        FastBatchNorm1d(last_mlp_opt.nn[i], momentum=last_mlp_opt.bn_momentum),
                        LeakyReLU(0.2),
                    ]
                ),
            )
            in_feat = last_mlp_opt.nn[i]

        if last_mlp_opt.dropout:
            self.FC_layer.add_module("Dropout", Dropout(p=last_mlp_opt.dropout))

        self.FC_layer.add_module("Last", Lin(in_feat, self.out_channels, bias=False))
        self.mode = option.loss_mode
        self.normalize_feature = option.normalize_feature
        self.loss_names = ["loss_reg", "loss"]

        self.lambda_reg = self.get_from_opt(option, ["loss_weights", "lambda_reg"])
        if self.lambda_reg:
            self.loss_names += ["loss_regul"]

        self.lambda_internal_losses = self.get_from_opt(option, ["loss_weights", "lambda_internal_losses"])

        self.visual_names = ["data_visual"]

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        # data = data.to(device)
        if isinstance(data, PairMultiScaleBatch):
            self.pre_computed = [f.to(device) for f in data.multiscale]
            self.upsample = [f.to(device) for f in data.upsample]
        else:
            self.upsample = None
            self.pre_computed = None

        self.input = Data(pos=data.pos, x=data.x, batch=data.batch).to(device)
        if hasattr(data, "pos_target"):
            if isinstance(data, PairMultiScaleBatch):
                self.pre_computed_target = [f.to(device) for f in data.multiscale_target]
                self.upsample_target = [f.to(device) for f in data.upsample_target]
            else:
                self.upsample_target = None
                self.pre_computed_target = None
            self.input_target = Data(pos=data.pos_target, x=data.x_target, batch=data.batch_target).to(device)
            self.match = data.pair_ind.to(torch.long).to(device)
            self.size_match = data.size_pair_ind.to(torch.long).to(device)
        else:
            self.match = None

    def apply_nn(self, input, pre_computed, upsample):
        stack_down = []
        data = input
        for i in range(len(self.down_modules) - 1):
            data = self.down_modules[i](data, precomputed=pre_computed)
            stack_down.append(data)

        data = self.down_modules[-1](data, precomputed=pre_computed)
        innermost = False

        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append(data)
            data = self.inner_modules[0](data)
            innermost = True

        for i in range(len(self.up_modules)):
            if i == 0 and innermost:
                data = self.up_modules[i]((data, stack_down.pop()))
            else:
                data = self.up_modules[i]((data, stack_down.pop()), precomputed=upsample)

        output = self.FC_layer(data.x)
        if self.normalize_feature:
            return output / (torch.norm(output, p=2, dim=1, keepdim=True) + 1e-3)
        else:
            return output

    def forward(self, *args, **kwargs):
        self.output = self.apply_nn(self.input, self.pre_computed, self.upsample)
        if self.match is None:
            return self.output

        self.output_target = self.apply_nn(self.input_target, self.pre_computed_target, self.upsample_target)
        self.compute_loss()

        return self.output

    def compute_loss(self):
        self.loss = 0

        # Collect internal losses and set them with self and them to self for later tracking
        if self.lambda_internal_losses:
            self.loss_internal = self.collect_internal_losses(lambda_weight=self.lambda_internal_losses)
            self.loss += self.loss_internal

        if self.mode == "match":
            self.loss_reg = self.compute_loss_match()
        elif self.mode == "label":
            self.loss_reg = self.compute_loss_label()

        self.loss += self.loss_reg

    def get_batch(self):
        if self.match is not None:
            batch = self.input.batch
            batch_target = self.input_target.batch
            return batch, batch_target
        else:
            return None

    def get_input(self):
        if self.match is not None:
            input = Data(pos=self.input.pos, ind=self.match[:, 0], size=self.size_match)
            input_target = Data(pos=self.input_target.pos, ind=self.match[:, 1], size=self.size_match)
            return input, input_target
        else:
            input = Data(pos=self.xyz)
            return input
