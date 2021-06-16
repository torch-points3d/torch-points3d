import torch
import torch.nn as nn
from torch.nn import LeakyReLU, Linear, Sequential

from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn import voxel_grid
from torch_geometric.data import Batch

from torch_scatter import scatter_max, scatter_mean

from torch_points3d.applications.sparseconv3d import SparseConv3d
from torch_points3d.core.common_modules import FastBatchNorm1d, Seq
from torch_points3d.core.common_modules import MLP
from torch_points3d.models.registration.base import FragmentBaseModel


class UnetMSparseConv3d(nn.Module):
    def __init__(
        self,
        backbone,
        input_nc=1,
        grid_size=0.05,
        pointnet_nn=None,
        pre_mlp_nn=None,
        post_mlp_nn=[64, 64, 32],
        add_pos=False,
        add_pre_x=False,
        backend="minkowski",
        aggr=None,
    ):
        if input_nc is None:
            input_nc = 1
        nn.Module.__init__(self)
        self.unet = SparseConv3d(architecture="unet", input_nc=input_nc, config=backbone, backend=backend)
        if pre_mlp_nn is not None:
            self.pre_mlp = MLP(pre_mlp_nn)
        else:
            self.pre_mlp = torch.nn.Identity()
        if pointnet_nn is not None:
            self.pointnet = MLP(pointnet_nn)
        else:
            self.pointnet = torch.nn.Identity()
        self.post_mlp = MLP(post_mlp_nn)
        self._grid_size = grid_size
        self.add_pos = add_pos
        self.add_pre_x = add_pre_x
        self.aggr = aggr

    def set_grid_size(self, grid_size):
        self._grid_size = grid_size

    def _aggregate(self, x, cluster, unique_pos_indices):
        if self.aggr is None:
            return x[unique_pos_indices]
        elif self.aggr == "mean":
            return scatter_mean(x, cluster, dim=0)
        elif self.aggr == "max":
            res, _ = scatter_max(x, cluster, dim=0)
            return res
        else:
            raise NotImplementedError

    def _prepare_data(self, data):
        coords = torch.round((data.pos) / self._grid_size).long()
        cluster = voxel_grid(coords, data.batch, 1)
        cluster, unique_pos_indices = consecutive_cluster(cluster)

        coords = coords[unique_pos_indices]
        new_batch = data.batch[unique_pos_indices]
        new_pos = data.pos[unique_pos_indices]
        x = self._aggregate(data.x, cluster, unique_pos_indices)
        sparse_data = Batch(x=x, pos=new_pos, coords=coords, batch=new_batch)
        return sparse_data, cluster

    def forward(self, data, **kwargs):
        inp = data.clone()
        if self.add_pos:
            inp.x = self.pointnet(torch.cat([inp.x, data.pos - data.pos.mean(0)], 1))
        else:
            inp.x = self.pointnet(inp.x)
        pre_x = inp.x
        d, cluster = self._prepare_data(inp)
        d.x = self.pre_mlp(d.x)
        d = self.unet.forward(d)
        inp_post_mlp = d.x[cluster]
        if self.add_pos:
            inp_post_mlp = torch.cat([inp_post_mlp, data.pos - data.pos.mean(0)], 1)
        if self.add_pre_x:
            inp_post_mlp = torch.cat([inp_post_mlp, pre_x], 1)
        data.x = self.post_mlp(inp_post_mlp)
        return data


class BaseMS_SparseConv3d(FragmentBaseModel):
    def __init__(self, option, model_type, dataset, modules):
        FragmentBaseModel.__init__(self, option)
        self.mode = option.loss_mode
        self.normalize_feature = option.normalize_feature
        self.loss_names = ["loss_reg", "loss"]
        self.metric_loss_module, self.miner_module = FragmentBaseModel.get_metric_loss_and_miner(
            getattr(option, "metric_loss", None), getattr(option, "miner", None)
        )

    def set_input(self, data, device):
        self.input, self.input_target = data.to_data()
        self.input = self.input.to(device)
        if getattr(data, "pos_target", None) is not None:
            self.input_target = self.input_target.to(device)
            self.match = data.pair_ind.to(torch.long).to(device)
            self.size_match = data.size_pair_ind.to(torch.long).to(device)
        else:
            self.match = None

    def get_batch(self):
        if self.match is not None:
            return self.input.batch, self.input_target.batch
        else:
            return self.input.batch, None

    def get_input(self):
        if self.match is not None:
            input = self.input.clone()
            input_target = self.input_target.clone()
            input.ind = self.match[:, 0]
            input_target.ind = self.match[:, 1]
            input.size = self.size_match
            input_target.size = self.size_match
            return input, input_target
        else:
            return self.input, None

    def apply_nn(self, input):
        raise NotImplementedError("It depends on the networks")


class MS_SparseConv3d(BaseMS_SparseConv3d):
    def __init__(self, option, model_type, dataset, modules):
        # Last Layer
        BaseMS_SparseConv3d.__init__(self, option, model_type, dataset, modules)
        option_unet = option.option_unet
        num_scales = option_unet.num_scales
        self.unet = nn.ModuleList()
        for i in range(num_scales):
            module = UnetMSparseConv3d(
                option_unet.backbone,
                input_nc=option_unet.input_nc,
                grid_size=option_unet.grid_size[i],
                pointnet_nn=getattr(option_unet, "pointnet_nn", None),
                post_mlp_nn=getattr(option_unet, "post_mlp_nn", [64, 64, 32]),
                pre_mlp_nn=getattr(option_unet, "pre_mlp_nn", None),
                add_pos=getattr(option_unet, "add_pos", False),
                add_pre_x=getattr(option_unet, "add_pre_x", False),
                aggr=getattr(option_unet, "aggr", None),
                backend=option.backend,
            )
            self.unet.add_module(name=str(i), module=module)
        # Last MLP layer
        assert option.mlp_cls is not None
        last_mlp_opt = option.mlp_cls
        self.FC_layer = Seq()
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.append(
                Sequential(
                    *[
                        Linear(last_mlp_opt.nn[i - 1], last_mlp_opt.nn[i], bias=False),
                        FastBatchNorm1d(last_mlp_opt.nn[i], momentum=last_mlp_opt.bn_momentum),
                        LeakyReLU(0.2),
                    ]
                )
            )

    def apply_nn(self, input):
        # inputs = self.compute_scales(input)
        outputs = []
        for i in range(len(self.unet)):
            out = self.unet[i](input.clone())
            out.x = out.x / (torch.norm(out.x, p=2, dim=1, keepdim=True) + 1e-20)
            outputs.append(out)
        x = torch.cat([o.x for o in outputs], 1)
        out_feat = self.FC_layer(x)
        if self.normalize_feature:
            out_feat = out_feat / (torch.norm(out_feat, p=2, dim=1, keepdim=True) + 1e-20)
        return out_feat


class MS_SparseConv3d_Shared(BaseMS_SparseConv3d):
    def __init__(self, option, model_type, dataset, modules):
        BaseMS_SparseConv3d.__init__(self, option, model_type, dataset, modules)
        option_unet = option.option_unet
        self.grid_size = option_unet.grid_size
        self.unet = UnetMSparseConv3d(
            option_unet.backbone,
            input_nc=option_unet.input_nc,
            pointnet_nn=getattr(option_unet, "pointnet_nn", None),
            post_mlp_nn=getattr(option_unet, "post_mlp_nn", [64, 64, 32]),
            pre_mlp_nn=getattr(option_unet, "pre_mlp_nn", None),
            add_pos=getattr(option_unet, "add_pos", False),
            add_pre_x=getattr(option_unet, "add_pre_x", False),
            aggr=getattr(option_unet, "aggr", None),
            backend=option.backend,
        )
        assert option.mlp_cls is not None
        last_mlp_opt = option.mlp_cls
        self.FC_layer = Seq()
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.append(
                Sequential(
                    *[
                        Linear(last_mlp_opt.nn[i - 1], last_mlp_opt.nn[i], bias=False),
                        FastBatchNorm1d(last_mlp_opt.nn[i], momentum=last_mlp_opt.bn_momentum),
                        LeakyReLU(0.2),
                    ]
                )
            )

        # Intermediate loss
        if getattr(option, "intermediate_loss", None) is not None:
            int_loss_option = option.intermediate_loss
            self.int_metric_loss, _ = FragmentBaseModel.get_metric_loss_and_miner(
                getattr(int_loss_option, "metric_loss", None), getattr(int_loss_option, "miner", None)
            )
            self.int_weights = int_loss_option.weights
            for i in range(len(int_loss_option.weights)):
                self.loss_names += ["loss_intermediate_loss_{}".format(i)]
        else:
            self.int_metric_loss = None

    def compute_intermediate_loss(self, outputs, outputs_target):
        assert len(outputs) == len(outputs_target)
        if self.int_metric_loss is not None:
            assert len(outputs) == len(self.int_weights)
            for i, w in enumerate(self.int_weights):
                xyz = self.input.pos
                xyz_target = self.input_target.pos
                loss_i = self.int_metric_loss(outputs[i].x, outputs_target[i].x, self.match[:, :2], xyz, xyz_target)
                self.loss += w * loss_i
                setattr(self, "loss_intermediate_loss_{}".format(i), loss_i)

    def apply_nn(self, input):
        # inputs = self.compute_scales(input)
        outputs = []
        for i in range(len(self.grid_size)):
            self.unet.set_grid_size(self.grid_size[i])
            out = self.unet(input.clone())
            out.x = out.x / (torch.norm(out.x, p=2, dim=1, keepdim=True) + 1e-20)
            outputs.append(out)
        x = torch.cat([o.x for o in outputs], 1)
        out_feat = self.FC_layer(x)
        if self.normalize_feature:
            out_feat = out_feat / (torch.norm(out_feat, p=2, dim=1, keepdim=True) + 1e-20)
        return out_feat, outputs

    def forward(self, *args, **kwargs):
        self.output, outputs = self.apply_nn(self.input)
        if self.match is None:
            return self.output

        self.output_target, outputs_target = self.apply_nn(self.input_target)
        self.compute_loss()

        self.compute_intermediate_loss(outputs, outputs_target)

        return self.output


class MS_SparseConv3d_Shared_Pool(MS_SparseConv3d_Shared):
    def __init__(self, option, model_type, dataset, modules):

        MS_SparseConv3d_Shared.__init__(self, option, model_type, dataset, modules)
        self.pool_mode = option.pool_mode

    def apply_nn(self, input):
        # inputs = self.compute_scales(input)
        outputs = []
        for i in range(len(self.grid_size)):
            self.unet.set_grid_size(self.grid_size[i])
            out = self.unet(input.clone())
            out.x = out.x / (torch.norm(out.x, p=2, dim=1, keepdim=True) + 1e-20)
            outputs.append(out)

        if self.pool_mode == "max":
            x = torch.cat([o.x.unsqueeze(0) for o in outputs], 0).max(0)[0]
        elif self.pool_mode == "mean":
            x = torch.cat([o.x.unsqueeze(0) for o in outputs], 0).mean(0)
        else:
            raise NotImplementedError

        out_feat = self.FC_layer(x)
        if self.normalize_feature:
            out_feat = out_feat / (torch.norm(out_feat, p=2, dim=1, keepdim=True) + 1e-20)
        return out_feat, outputs
