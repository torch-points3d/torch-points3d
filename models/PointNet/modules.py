
import torch 
from torch_geometric.nn import global_max_pool

from models.core_modules import * 

class MiniPointNet(torch.nn.Module):

    def __init__(self, local_nn, global_nn):
        super(STN, self).__init__()

        self.local_nn = MLP(local_nn)
        self.global_nn = MLP(global_nn)

    def forward(self, data):
        
        x = data.x
        x = self.local_nn(x)
        x = global_max_pool(x, data.batch)
        x = self.global_nn(x)

        return x

class STN(torch.nn.Module):

    def 