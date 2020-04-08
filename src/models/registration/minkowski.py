import logging
import torch

from src.modules.MinkowskiEngine import *
from src.models.base_architectures import UnwrappedUnetBasedModel
from src.models.base_model import BaseModel

from torch.nn import Sequential, Linear, LeakyReLU, Dropout
from src.core.common_modules import FastBatchNorm1d

log = logging.getLogger(__name__)


class Minkowski_Baseline_Model_Fragment(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        super(Minkowski_Baseline_Model_Fragment, self).__init__(option)
        self.model = initialize_minkowski_unet(
            option.model_name, dataset.feature_dimension, option.out_channels, option.D
        )
        self.metric_loss_module, self.miner_module = BaseModel.get_metric_loss_and_miner(
            getattr(option, "metric_loss", None), getattr(option, "miner", None)
        )
        self.num_pos_pairs = option.num_pos_pairs
        self.loss_names = ["loss_reg", "loss", "internal"]

        # Last Layer
        last_mlp_opt = option.mlp_cls
        in_feat = last_mlp_opt.nn[0]
        self.FC_layer = Sequential()
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.add_module(
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
            self.FC_layer.add_module("Dropout", Dropout(p=last_mlp_opt.dropout))

        self.FC_layer.add_module("Class", Linear(in_feat, in_feat, bias=False))

    def set_input(self, data, device):

        self.batch_idx = data.batch.squeeze()
        coords = torch.cat([data.batch.unsqueeze(-1).int(), data.pos.int()], -1)
        self.input = ME.SparseTensor(data.x, coords=coords).to(device)

        if hasattr(data, "pos_target"):
            coords_target = torch.cat([data.batch_target.unsqueeze(-1).int(), data.pos_target.int()], -1)
            self.input_target = ME.SparseTensor(data.x_target, coords=coords_target).to(device)
            num_pos_pairs = len(data.pair_ind)
            if self.num_pos_pairs < len(data.pair_ind):
                num_pos_pairs = self.num_pos_pairs
            rand_ind = self.randperm(len(data.pair_ind))[:num_pos_pairs]
            self.ind = data.pair_ind[rand_ind:, 0].to(device)
            self.ind_target = data.pair_ind[rand_ind:, 1].to(device)
            rang = torch.range(0, self.num_pos_pairs)
            self.labels = torch.cat([rang, rang], 0).to(device)
        else:
            self.labels = None

    def apply_nn(self, input):
        output = self.model(self.input)
        return ME.SparseTensor(
            output.F / (torch.norm(output.F, p=2, dim=1, keepdim=True) + 1e-3),
            coords_key=output.coords_key,
            coords_manager=output.coords_man,
        )

    def compute_loss(self):

        # miner
        hard_pairs = None
        if self.miner_module is not None:
            hard_pairs = self.miner_module(self.output, self.labels)

        # loss
        self.loss_reg = self.metric_loss_module(self.output, self.labels, hard_pairs)
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
            return self.output

    def backward(self):
        self.loss.backward()


class MinkowskiFragment(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnetBasedModel
        self.num_pos_pairs = option.num_pos_pairs
        self.normalize_feature = option.normalize_feature
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)
        self.loss_names = ["loss_reg", "loss"]
        # Last Layer
        last_mlp_opt = option.mlp_cls
        in_feat = last_mlp_opt.nn[0]
        self.FC_layer = Sequential()
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.add_module(
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
            self.FC_layer.add_module("Dropout", Dropout(p=last_mlp_opt.dropout))

        self.FC_layer.add_module("Class", Linear(in_feat, in_feat, bias=False))

    def set_input(self, data, device):
        coords = torch.cat([data.batch.unsqueeze(-1).int(), data.pos.int()], -1)
        self.input = ME.SparseTensor(data.x, coords=coords).to(device)

        if hasattr(data, "pos_target"):
            coords_target = torch.cat([data.batch_target.unsqueeze(-1).int(), data.pos_target.int()], -1)
            self.input_target = ME.SparseTensor(data.x_target, coords=coords_target).to(device)

            num_pos_pairs = len(data.pair_ind)
            if self.num_pos_pairs < len(data.pair_ind):
                num_pos_pairs = self.num_pos_pairs
            rand_ind = torch.randperm(len(data.pair_ind))[:num_pos_pairs]
            self.ind = data.pair_ind[rand_ind, 0].to(torch.long).to(device)
            self.ind_target = data.pair_ind[rand_ind, 1].to(torch.long).to(device)
            rang = torch.arange(0, num_pos_pairs)
            self.labels = torch.cat([rang, rang], 0).to(device)
        else:
            self.labels = None

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
            return ME.SparseTensor(
                out_feat / (torch.norm(out_feat, p=2, dim=1, keepdim=True) + 1e-5),
                coords_key=x.coords_key,
                coords_manager=x.coords_man,
            )
        else:
            return x

    def compute_loss(self):
        # miner

        hard_pairs = None
        if self.miner_module is not None:
            hard_pairs = self.miner_module(self.output, self.labels)
        # loss
        self.loss_reg = self.metric_loss_module(self.output, self.labels, hard_pairs)
        self.internal = self.get_internal_loss()
        self.loss = self.loss_reg + self.internal

    def forward(self):
        out = self.apply_nn(self.input)
        if self.labels is None:
            self.output = out.F
            return self.output

        out_target = self.apply_nn(self.input_target)
        self.output = torch.cat([out.F[self.ind], out_target.F[self.ind_target]], 0)
        self.compute_loss()
        return self.output

    def backward(self):
        if hasattr(self, "loss"):
            self.loss.backward()