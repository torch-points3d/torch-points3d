from torch_geometric.data import Data
import logging

from torch_points3d.modules.pointnet2 import *
from torch_points3d.core.common_modules.dense_modules import Conv1D
from torch_points3d.core.common_modules.base_modules import Seq
from torch_points3d.core.base_conv.dense import DenseFPModule
from torch_points3d.models.base_architectures import BackboneBasedModel
from torch_points3d.models.registration.base import create_batch_siamese

log = logging.getLogger(__name__)


class SiamesePointNet2_D(BackboneBasedModel):
    r"""
    PointNet2 with multi-scale grouping
    metric learning siamese network that uses feature propogation layers
    """

    def __init__(self, option, model_type, dataset, modules):
        BackboneBasedModel.__init__(self, option, model_type, dataset, modules)

        # Last MLP
        last_mlp_opt = option.mlp_cls
        self._dim_output = last_mlp_opt.nn[-1]

        self.FC_layer = Seq()
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.append(Conv1D(last_mlp_opt.nn[i - 1], last_mlp_opt.nn[i], bn=True, bias=False))

        self.loss_names = ["loss_patch_desc"]

    def set_input(self, data, device):
        assert len(data.pos.shape) == 3
        data = data.to(device)
        self.input = Data(x=data.x.transpose(1, 2).contiguous(), pos=data.pos)

    def forward(self, *args, **kwargs):
        r"""
        forward pass of the network
        """
        data = self.input
        for i in range(len(self.down_modules)):
            data = self.down_modules[i](data)
        last_feature = data.x
        self.output = self.FC_layer(last_feature).transpose(1, 2).contiguous().view((-1, self._dim_output))

        self.loss_reg = self.loss_module(self.output) + self.get_internal_loss()

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_reg.backward()  # calculate gradients of network G w.r.t. loss_G
