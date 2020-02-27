import torch
import etw_pytorch_utils as pt_utils
from torch_geometric.data import Data
import logging

from src.core.losses import *
from src.modules.pointnet2 import *
from src.core.base_conv.dense import DenseFPModule
from src.models.base_architectures import BackboneBasedModel
from src.models.registration.base import create_batch_siamese

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
        self.FC_layer = pt_utils.Seq(last_mlp_opt.nn[0])
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.conv1d(last_mlp_opt.nn[i], bn=True)

    def set_input(self, data):
        # Size : B x N x 3
        assert len(data.pos.shape) == 3
        if data.x is not None:
            x = data.x.transpose(1, 2).contiguous()
        else:
            x = None
        self.input = Data(x=x, pos=data.pos)
        self.labels = torch.range(0, data.pos.shape[0]).repeat(2, 1).T.reshape(-1)

    def forward(self):
        r"""
        forward pass of the network
        """
        data = self.input
        for i in range(len(self.down_modules)):
            data = self.down_modules[i](data)
        # size after pointnet B x N x D
        last_feature = torch.mean(data.x, dim=1)
        # size after global pooling B x D
        self.output = self.FC_layer(last_feature)
        hard_pairs = None
        if self.miner_module is not None:
            hard_pairs = self.miner_module(self.output, self.labels)
        self.loss_reg = self.loss_module(self.output, self.labels, hard_pairs)
        self.internal = self.get_internal_loss()
        self.loss = self.loss_reg + self.internal
        return self.output

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss.backward()  # calculate gradients of network G w.r.t. loss_G
