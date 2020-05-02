import logging
import torch

from torch_points3d.modules.MinkowskiEngine import *
from torch_points3d.models.base_architectures import UnwrappedUnetBasedModel
from torch_points3d.models.base_model import BaseModel

from torch.nn import Sequential, Linear, LeakyReLU, Dropout
from torch_points3d.core.common_modules import FastBatchNorm1d, Seq

log = logging.getLogger(__name__)


class BaseMinkowski(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        BaseModel.__init__(self, option)
        self.mode = option.loss_mode
        self.normalize_feature = option.normalize_feature
        self.loss_names = ["loss_reg", "loss"]
        self.metric_loss_module, self.miner_module = BaseModel.get_metric_loss_and_miner(
            getattr(option, "metric_loss", None), getattr(option, "miner", None)
        )
        # Last Layer

        if option.mlp_cls is not None:
            last_mlp_opt = option.mlp_cls
            in_feat = last_mlp_opt.nn[0]
            self.FC_layer = Seq()
            for i in range(1, len(last_mlp_opt.nn)):
                self.FC_layer.append(
                    str(i),
                    Sequential(
                        *[
                            Linear(in_feat, last_mlp_opt.nn[i], bias=False),
                            FastBatchNorm1d(last_mlp_opt.nn[i], momentum=last_mlp_opt.bn_momentum),
                            LeakyReLU(0.2),
                        ]
                    ),
                )
                in_feat = last_mlp_opt.nn[i]

            if last_mlp_opt.dropout:
                self.FC_layer.append(Dropout(p=last_mlp_opt.dropout))

            self.FC_layer.append(Linear(in_feat, in_feat, bias=False))
        else:
            self.FC_layer = torch.nn.Identity()

    def set_input(self, data, device):
        coords = torch.cat([data.batch.unsqueeze(-1).int(), data.pos.int()], -1)
        self.input = ME.SparseTensor(data.x, coords=coords).to(device)
        self.xyz = torch.stack((data.pos_x, data.pos_y, data.pos_z), 0).T.to(device)
        if hasattr(data, "pos_target"):
            coords_target = torch.cat([data.batch_target.unsqueeze(-1).int(), data.pos_target.int()], -1)
            self.input_target = ME.SparseTensor(data.x_target, coords=coords_target).to(device)
            self.xyz_target = torch.stack((data.pos_x_target, data.pos_y_target, data.pos_z_target), 0).T.to(device)
            self.match = data.pair_ind.to(torch.long).to(device)
            self.size_match = data.size_pair_ind.to(torch.long).to(device)
        else:
            self.match = None

    def compute_loss_match(self):
        self.loss_reg = self.metric_loss_module(
            self.output, self.output_target, self.match[:, :2], self.xyz, self.xyz_target
        )
        self.loss = self.loss_reg

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
        self.loss_reg = self.metric_loss_module(output, labels, hard_pairs)
        self.loss = self.loss_reg

    def apply_nn(self, input):
        raise NotImplementedError("Model still not defined")

    def forward(self):
        self.output = self.apply_nn(self.input)
        if self.match is None:
            return self.output

        self.output_target = self.apply_nn(self.input_target)
        if self.mode == "match":
            self.compute_loss_match()
        elif self.mode == "label":
            self.compute_loss_label()
        else:
            raise NotImplementedError("The mode for the loss is incorrect")

        return self.output

    def backward(self):
        if hasattr(self, "loss"):
            self.loss.backward()

    def get_outputs(self):
        if self.match is not None:
            return self.output, self.output_target
        else:
            return self.output

    def get_ind(self):
        if self.match is not None:
            return self.match[:, 0], self.match[:, 1], self.size_match
        else:
            return None

    def get_xyz(self):
        if self.match is not None:
            return self.xyz, self.xyz_target
        else:
            return self.xyz

    def get_batch_idx(self):
        if self.match is not None:
            batch = self.input.C[:, 0]
            batch_target = self.input_target.C[:, 0]
            return batch, batch_target
        else:
            return None


class Minkowski_Baseline_Model_Fragment(BaseMinkowski):
    def __init__(self, option, model_type, dataset, modules):
        BaseMinkowski.__init__(self, option, model_type, dataset, modules)

        self.model = initialize_minkowski_unet(
            option.model_name,
            in_channels=dataset.feature_dimension,
            out_channels=option.out_channels,
            D=option.D,
            conv1_kernel_size=option.conv1_kernel_size,
        )

    def apply_nn(self, input):
        output = self.model(input).F
        output = self.FC_layer(output)
        if self.normalize_feature:
            return output / (torch.norm(output, p=2, dim=1, keepdim=True) + 1e-3)
        else:
            return output


class MinkowskiFragment(BaseMinkowski, UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)
        self.mode = option.loss_mode
        self.normalize_feature = option.normalize_feature
        self.loss_names = ["loss_reg", "loss"]
        self.metric_loss_module, self.miner_module = BaseModel.get_metric_loss_and_miner(
            getattr(option, "metric_loss", None), getattr(option, "miner", None)
        )
        # Last Layer

        if option.mlp_cls is not None:
            last_mlp_opt = option.mlp_cls
            in_feat = last_mlp_opt.nn[0]
            self.FC_layer = Seq()
            for i in range(1, len(last_mlp_opt.nn)):
                self.FC_layer.append(
                    str(i),
                    Sequential(
                        *[
                            Linear(in_feat, last_mlp_opt.nn[i], bias=False),
                            FastBatchNorm1d(last_mlp_opt.nn[i], momentum=last_mlp_opt.bn_momentum),
                            LeakyReLU(0.2),
                        ]
                    ),
                )
                in_feat = last_mlp_opt.nn[i]

            if last_mlp_opt.dropout:
                self.FC_layer.append(Dropout(p=last_mlp_opt.dropout))

            self.FC_layer.append(Linear(in_feat, in_feat, bias=False))
        else:
            self.FC_layer = torch.nn.Identity()

    def apply_nn(self, input):
        x = input
        stack_down = []
        for i in range(len(self.down_modules) - 1):
            x = self.down_modules[i](x)
            stack_down.append(x)

        x = self.down_modules[-1](x)
        stack_down.append(None)

        for i in range(len(self.up_modules)):
            x = self.up_modules[i](x, stack_down.pop())
        out_feat = self.FC_layer(x.F)
        # out_feat = x.F
        if self.normalize_feature:
            return out_feat / (torch.norm(out_feat, p=2, dim=1, keepdim=True) + 1e-20)
        else:
            return out_feat
