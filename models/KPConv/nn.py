import torch
torch.backends.cudnn.enabled = False
import numpy as np
import torch.nn.functional as F
from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import knn_interpolate
from torch_geometric.utils import intersection_and_union as i_and_u
from .modules import MLP, PointKernel, KPConv

class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip

class PartSegmentation(torch.nn.Module):
    def __init__(self, num_classes):
        super(PartSegmentation, self).__init__()
        self.kp1_module = KPConv(0.3, 0.2, 3, 32)
        self.kp2_module = KPConv(0.5, 0.4, 32, 64)

        self.fp2_module = FPModule(3, MLP([64 + 32, 32]))
        self.fp1_module = FPModule(3, MLP([32, 32, 32]))

        self.mlp_cls = MLP([32, 16, num_classes])

    def forward(self, data):
        #Normalize in [-.5, .5]
        max_, min_ = np.max(data.pos.cpu().numpy()), np.min(data.pos.cpu().numpy())
        data.pos = (data.pos - (max_ + min_) / 2.) / np.linalg.norm(max_ - min_)
        
        input = (data.x, data.pos, data.batch)
        kp1_out = self.kp1_module(*input)
        kp2_out = self.kp2_module(*kp1_out)

        fp2_out = self.fp2_module(*kp2_out, *kp1_out)
        x, _, _ = self.fp1_module(*fp2_out, *input)
        x = self.mlp_cls(x)
        return F.log_softmax(x, dim=-1)