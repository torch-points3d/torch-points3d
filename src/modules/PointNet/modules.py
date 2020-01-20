import torch
from torch.nn import Linear
from torch_geometric.nn import global_max_pool

from models.core_modules import *
from models.core_transforms import BaseLinearTransformSTNkD
from models.base_model import BaseInternalLossModule


class MiniPointNet(torch.nn.Module):
    def __init__(self, local_nn, global_nn):
        super().__init__()

        self.local_nn = MLP(local_nn)
        self.global_nn = MLP(global_nn)

    def forward(self, x, batch):
        x = self.local_nn(x)
        x = global_max_pool(x, batch)
        x = self.global_nn(x)

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
        self._batch_size = None

    def set_scatter_pooling(self, use_scatter_pooling, batch_size):
        self._use_scatter_pooling = use_scatter_pooling
        self._batch_size = batch_size

    def func_global_max_pooling(self, x3, batch):
        if self._use_scatter_pooling:
            return global_max_pool(x3, batch)
        else:
            global_feature = x3.view((self._batch_size, -1, x3.shape[-1])).max(1)
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
