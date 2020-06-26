import logging
import torch
from torch_geometric.data import Data

from torch_points3d.modules.MinkowskiEngine import *
from torch_points3d.models.base_architectures import UnwrappedUnetBasedModel
from torch_points3d.models.registration.base import FragmentBaseModel
from torch.nn import Sequential, Linear, LeakyReLU, Dropout
from torch_points3d.core.common_modules import FastBatchNorm1d, Seq


log = logging.getLogger(__name__)


class BaseMinkowski(FragmentBaseModel):
    def __init__(self, option, model_type, dataset, modules):
        FragmentBaseModel.__init__(self, option)
        self.mode = option.loss_mode
        self.normalize_feature = option.normalize_feature
        self.loss_names = ["loss_reg", "loss"]
        self.metric_loss_module, self.miner_module = FragmentBaseModel.get_metric_loss_and_miner(
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
        coords = torch.cat([data.batch.unsqueeze(-1).int(), data.coords.int()], -1)
        self.input = ME.SparseTensor(data.x, coords=coords).to(device)
        self.xyz = data.pos.to(device)
        if hasattr(data, "pos_target"):
            coords_target = torch.cat([data.batch_target.unsqueeze(-1).int(), data.coords_target.int()], -1)
            self.input_target = ME.SparseTensor(data.x_target, coords=coords_target).to(device)
            self.xyz_target = data.pos_target.to(device)
            self.match = data.pair_ind.to(torch.long).to(device)
            self.size_match = data.size_pair_ind.to(torch.long).to(device)
        else:
            self.match = None

    def get_batch(self):
        if self.match is not None:
            batch = self.input.C[:, 0]
            batch_target = self.input_target.C[:, 0]
            return batch, batch_target
        else:
            return None, None

    def get_input(self):
        if self.match is not None:
            input = Data(pos=self.xyz, ind=self.match[:, 0], size=self.size_match)
            input_target = Data(pos=self.xyz_target, ind=self.match[:, 1], size=self.size_match)
            return input, input_target
        else:
            input = Data(pos=self.xyz)
            return input, None


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
        self.metric_loss_module, self.miner_module = FragmentBaseModel.get_metric_loss_and_miner(
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
