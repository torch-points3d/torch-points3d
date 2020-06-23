import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch.nn import Sequential
from torch_geometric.data import Data
import logging

from torch_points3d.core.losses import *
from torch_points3d.modules.pointnet2 import *
from torch_points3d.core.common_modules import MLP
from torch_points3d.models.base_architectures import BackboneBasedModel
from torch_points3d.models.base_architectures import UnetBasedModel
from torch_points3d.core.common_modules.dense_modules import Conv1D
from torch_points3d.core.common_modules.base_modules import Seq
from torch_points3d.models.registration.base import FragmentBaseModel

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
            self.FC_layer = Sequential(Lin(last_mlp_opt.nn[-2], last_mlp_opt.nn[-1]))
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

    def forward(self, *args, **kwargs):
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


class FragmentPointNet2_D(UnetBasedModel, FragmentBaseModel):

    r"""
        PointNet2 with multi-scale grouping
        descriptors network for registration that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnetBasedModel
        UnetBasedModel.__init__(self, option, model_type, dataset, modules)
        # Last MLP
        self.mode = option.loss_mode
        self.normalize_feature = option.normalize_feature
        self.out_channels = option.out_channels
        self.loss_names = ["loss_reg", "loss"]
        self.metric_loss_module, self.miner_module = UnetBasedModel.get_metric_loss_and_miner(
            getattr(option, "metric_loss", None), getattr(option, "miner", None)
        )
        last_mlp_opt = option.mlp_cls

        self.FC_layer = Seq()
        last_mlp_opt.nn[0]
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.append(Conv1D(last_mlp_opt.nn[i - 1], last_mlp_opt.nn[i], bn=True, bias=False))
        if last_mlp_opt.dropout:
            self.FC_layer.append(torch.nn.Dropout(p=last_mlp_opt.dropout))

        self.FC_layer.append(Conv1D(last_mlp_opt.nn[-1], self.out_channels, activation=None, bias=True, bn=False))

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        Sets:
            self.input:
                x -- Features [B, C, N]
                pos -- Points [B, N, 3]
        """
        assert len(data.pos.shape) == 3

        if data.x is not None:
            x = data.x.transpose(1, 2).contiguous()
        else:
            x = None
        self.input = Data(x=x, pos=data.pos).to(device)

        if hasattr(data, "pos_target"):
            if data.x_target is not None:
                x = data.x_target.transpose(1, 2).contiguous()
            else:
                x = None
            self.input_target = Data(x=x, pos=data.pos_target).to(device)
            self.match = data.pair_ind.to(torch.long).to(device)
            self.size_match = data.size_pair_ind.to(torch.long).to(device)
        else:
            self.match = None

    def apply_nn(self, input):
        last_feature = self.model(input).x
        output = self.FC_layer(last_feature).transpose(1, 2).contiguous().view((-1, self.out_channels))
        if self.normalize_feature:
            return output / (torch.norm(output, p=2, dim=1, keepdim=True) + 1e-5)
        else:
            return output

    def get_input(self):
        if self.match is not None:
            input = Data(pos=self.input.pos.view(-1, 3), ind=self.match[:, 0], size=self.size_match)
            input_target = Data(pos=self.input_target.pos.view(-1, 3), ind=self.match[:, 1], size=self.size_match)
            return input, input_target
        else:
            input = Data(pos=self.input.pos.view(-1, 3))
            return input

    def get_batch(self):
        if self.match is not None:
            batch = (
                torch.arange(0, self.input.pos.shape[0])
                .view(-1, 1)
                .repeat(1, self.input.pos.shape[1])
                .view(-1)
                .to(self.input.pos.device)
            )
            batch_target = (
                torch.arange(0, self.input_target.pos.shape[0])
                .view(-1, 1)
                .repeat(1, self.input_target.pos.shape[1])
                .view(-1)
                .to(self.input.pos.device)
            )
            return batch, batch_target
        else:
            return None, None
