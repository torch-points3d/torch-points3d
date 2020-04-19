import torch
from torch_geometric.nn import global_max_pool, global_mean_pool

from torch_points3d.core.common_modules.base_modules import *
from torch_points3d.core.common_modules.spatial_transform import BaseLinearTransformSTNkD
from torch_points3d.models.base_model import BaseInternalLossModule


class MiniPointNet(torch.nn.Module):
    def __init__(self, local_nn, global_nn, aggr="max", return_local_out=False):
        super().__init__()

        self.local_nn = MLP(local_nn)
        self.global_nn = MLP(global_nn) if global_nn else None
        self.g_pool = global_max_pool if aggr == "max" else global_mean_pool
        self.return_local_out = return_local_out

    def forward(self, x, batch):
        y = x = self.local_nn(x)  # [num_points, in_dim] -> [num_points, local_out_nn]
        x = self.g_pool(x, batch)  # [num_points, local_out_nn] -> [local_out_nn]
        if self.global_nn:
            x = self.global_nn(x)  # [local_out_nn] -> [global_out_nn]
        if self.return_local_out:
            return x, y
        return x

    def forward_embedding(self, pos, batch):
        global_feat, local_feat = self.forward(pos, batch)
        indices = batch.unsqueeze(-1).repeat((1, global_feat.shape[-1]))
        gathered_global_feat = torch.gather(global_feat, 0, indices)
        x = torch.cat([local_feat, gathered_global_feat], -1)
        return x


class PointNetSTN3D(BaseLinearTransformSTNkD):
    def __init__(self, local_nn=[3, 64, 128, 1024], global_nn=[1024, 512, 256], batch_size=1):
        super().__init__(MiniPointNet(local_nn, global_nn), global_nn[-1], 3, batch_size)

    def forward(self, x, batch):
        return super().forward(x, x, batch)


class PointNetSTNkD(BaseLinearTransformSTNkD, BaseInternalLossModule):
    def __init__(self, k=64, local_nn=[64, 64, 128, 1024], global_nn=[1024, 512, 256], batch_size=1):
        super().__init__(MiniPointNet(local_nn, global_nn), global_nn[-1], k, batch_size)

    def forward(self, x, batch):
        return super().forward(x, x, batch)

    def get_internal_losses(self):
        return {"orthogonal_regularization_loss": self.get_orthogonal_regularization_loss()}


class PointNetSeg(torch.nn.Module):
    def __init__(
        self,
        input_stn_local_nn=[3, 64, 128, 1024],
        input_stn_global_nn=[1024, 512, 256],
        local_nn_1=[3, 64, 64],
        feat_stn_k=64,
        feat_stn_local_nn=[64, 64, 128, 1024],
        feat_stn_global_nn=[1024, 512, 256],
        local_nn_2=[64, 64, 128, 1024],
        seg_nn=[1088, 512, 256, 128, 4],
        batch_size=1,
        *args,
        **kwargs
    ):
        super().__init__()

        self.batch_size = batch_size

        self.input_stn = PointNetSTN3D(input_stn_local_nn, input_stn_global_nn, batch_size)
        self.local_nn_1 = MLP(local_nn_1)
        self.feat_stn = PointNetSTNkD(feat_stn_k, feat_stn_local_nn, feat_stn_global_nn, batch_size)
        self.local_nn_2 = MLP(local_nn_2)
        self.seg_nn = MLP(seg_nn)

        self._use_scatter_pooling = True

    def set_scatter_pooling(self, use_scatter_pooling):
        self._use_scatter_pooling = use_scatter_pooling

    def func_global_max_pooling(self, x3, batch):
        if self._use_scatter_pooling:
            return global_max_pool(x3, batch)
        else:
            global_feature = x3.max(1)
            return global_feature[0]

    def forward(self, x, batch):

        # apply pointnet classification network to get per-point
        # features and global feature
        x = self.input_stn(x, batch)
        x = self.local_nn_1(x)
        x_feat_trans = self.feat_stn(x, batch)
        x3 = self.local_nn_2(x_feat_trans)

        global_feature = self.func_global_max_pooling(x3, batch)
        # concat per-point and global feature and regress to get
        # per-point scores
        feat_concat = torch.cat([x_feat_trans, global_feature[batch]], dim=1)
        out = self.seg_nn(feat_concat)

        return out
