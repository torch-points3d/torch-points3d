import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch.nn import Sequential as Seq
import logging

from torch_points3d.core.losses import *
from torch_points3d.modules.pointnet2 import *
from torch_points3d.core.base_conv.dense import DenseFPModule
from torch_points3d.core.common_modules import MLP
from torch_points3d.models.base_architectures import BackboneBasedModel
from torch_points3d.models.registration.base import create_batch_siamese
from torch_points3d.datasets.registration.pair import PairMultiScaleBatch
from torch_geometric.nn import global_mean_pool

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

    def forward(self) -> Any:

        self.output = self.apply_nn(self.input, self.pre_computed, self.batch_idx)
        if self.labels is None:
            return self.output
        else:
            output_target = self.apply_nn(self.input_target, self.pre_computed_target, self.batch_idx_target)
            self.output = torch.cat([self.output, output_target], 0)
            self.compute_loss()
            return self.output

    def compute_loss(self):
        if self._weight_classes is not None:
            self._weight_classes = self._weight_classes.to(self.output.device)

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
