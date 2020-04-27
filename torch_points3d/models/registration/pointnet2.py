import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch.nn import Sequential as Seq
from torch_geometric.data import Data
import logging

from torch_points3d.core.losses import *
from torch_points3d.modules.pointnet2 import *
from torch_points3d.core.base_conv.dense import DenseFPModule
from torch_points3d.core.common_modules import MLP
from torch_points3d.models.base_architectures import BackboneBasedModel
from torch_points3d.models.base_architectures import UnwrappedUnetBasedModel
from torch_points3d.models.registration.base import create_batch_siamese

log = logging.getLogger(__name__)


class PatchPointNet2_D(BackboneBasedModel):
    r"""
    PointNet2 with multi-scale grouping
    metric learning siamese network that uses feature propogation layers


    """

    def __init__(self, option, model_type, dataset, modules):
        BackboneBasedModel.__init__(self, option, model_type, dataset, modules)

        # Last MLP
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
        # Size : B x N x 3
        # manually concatenate the b

        if getattr(data, "pos_target", None) is not None:
            assert len(data.pos.shape) == 3 and len(data.pos_target.shape) == 3
            if data.x is not None:
                x = torch.cat([data.x, data.x_target], 0)
            else:
                x = None
            pos = torch.cat([data.pos, data.pos_target], 0)
            rang = torch.arange(0, data.pos.shape[0])

            labels = torch.cat([rang, rang], 0)
        else:
            x = data.x
            pos = data.pos
            labels = None

        if x is not None:
            x = x.transpose(1, 2).contiguous()

        self.input = Data(x=x, pos=pos, y=labels).to(device)

    def forward(self):
        r"""
        forward pass of the network
        """
        data = self.input
        labels = data.y
        for i in range(len(self.down_modules)):
            data = self.down_modules[i](data)

        # size after pointnet B x D x N
        last_feature = torch.mean(data.x, dim=-1)
        # size after global pooling B x D
        self.output = self.FC_layer(last_feature)
        self.output = F.normalize(self.output, p=2, dim=1)
        if labels is None:
            return self.output
        hard_pairs = None
        if self.miner_module is not None:
            hard_pairs = self.miner_module(self.output, labels)
        self.loss_reg = self.metric_loss_module(self.output, labels, hard_pairs)
        self.internal = self.get_internal_loss()
        self.loss = self.loss_reg + self.internal
        return self.output

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        if hasattr(self, "loss"):
            self.loss.backward()  # calculate gradients of network G w.r.t. loss_G


class FragmentPointNet(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)
        self.set_last_mlp(option.mlp_cls)
        self.loss_names = ["loss_reg", "loss", "internal"]
