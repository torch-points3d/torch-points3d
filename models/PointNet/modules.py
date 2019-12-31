
import torch 
from torch.nn import Linear
from torch_geometric.nn import global_max_pool

from models.core_modules import * 

class MiniPointNet(torch.nn.Module):

    def __init__(self, local_nn, global_nn):
        super(MiniPointNet, self).__init__()

        self.local_nn = MLP(local_nn)
        self.global_nn = MLP(global_nn)

    def forward(self, x, batch):
        
        x = self.local_nn(x)
        x = global_max_pool(x, batch)
        x = self.global_nn(x)

        return x

class AffineSTNkD(torch.nn.Module):

    def __init__(self, dim=3, local_nn=[3, 64, 128, 1024], global_nn=[1024, 512, 256], num_batches=1):
        super(AffineSTNkD, self).__init__()

        self.dim = dim 
        self.num_batches = num_batches

        self.mini_point_net = MiniPointNet(local_nn, global_nn)

        self.fc_layer = Linear(global_nn[-1], dim*dim)
        torch.nn.init.constant_(self.fc_layer.weight, 0)
        torch.nn.init.constant_(self.fc_layer.bias, 0)

        self.identity = torch.eye(dim).view(1, dim*dim)

    def forward(self, x, batch):

        global_feat = self.mini_point_net(x, batch)
        trans = self.fc_layer(global_feat)

        iden = self.identity.to(x.device).repeat(len(trans), 1)
        trans = trans + iden

        trans = trans.view(-1, self.dim, self.dim)

        batch_x = x.view(self.num_batches, -1, x.shape[1])

        trans_x = torch.bmm(batch_x, trans)

        return trans_x.view(len(x), x.shape[1])

