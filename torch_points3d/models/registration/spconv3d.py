import torch

from torch_points3d.models.registration.base import FragmentBaseModel
from torch.nn import Sequential, Linear, LeakyReLU, Dropout
from torch_geometric.data import Data, Batch
from torch_points3d.core.common_modules import FastBatchNorm1d, Seq
from torch_points3d.applications.sparseconv3d import SparseConv3d


class SparseConv3D(FragmentBaseModel):
    def __init__(self, option, model_type, dataset, modules):
        FragmentBaseModel.__init__(self, option)
        self.mode = option.loss_mode
        self.normalize_feature = option.normalize_feature
        self.loss_names = ["loss_reg", "loss"]
        self.metric_loss_module, self.miner_module = FragmentBaseModel.get_metric_loss_and_miner(
            getattr(option, "metric_loss", None), getattr(option, "miner", None)
        )
        # Unet
        self.backbone = SparseConv3d(
            "unet", dataset.feature_dimension, config=option.backbone, backend=option.get("backend", "minkowski")
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
        self.input = Batch(pos=data.pos, x=data.x, batch=data.batch).to(device)
        if hasattr(data, "pos_target"):
            self.input_target = Batch(pos=data.pos_target, x=data.x_target, batch=data.batch_target).to(device)
            self.match = data.pair_ind.to(torch.long).to(device)
            self.size_match = data.size_pair_ind.to(torch.long).to(device)
        else:
            self.match = data.pair_ind.to(torch.long).to(device)
            self.size_match = data.size_pair_ind.to(torch.long).to(device)

    def get_batch(self):
        if self.match is not None:
            batch = self.input.batch
            batch_target = self.input_target.batch
            return batch, batch_target
        else:
            return None, None

    def get_input(self):
        if self.match is not None:
            inp = Data(pos=self.input.pos, ind=self.match[:, 0], size=self.size_match)
            inp_target = Data(pos=self.input_target.pos, ind=self.match[:, 1], size=self.size_match)
            return inp, inp_target
        else:
            return self.input

    def apply_nn(self, input):

        out_feat = self.backbone(input).x
        out_feat = self.FC_layer(out_feat)
        if self.normalize_feature:
            return out_feat / (torch.norm(out_feat, p=2, dim=1, keepdim=True) + 1e-20)
        else:
            return out_feat
