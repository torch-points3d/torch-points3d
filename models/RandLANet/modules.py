
import torch 
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, knn
from models.base_model import *
import math

class RandlaConv(MessagePassing):
    '''
        Implements both the Local Spatial Encoding and Attentive Pooling blocks from 
        RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds
        https://arxiv.org/pdf/1911.11236


    '''

    def __init__(self, ratio=None, k=None, point_pos_nn=None, attention_nn=None, global_nn=None, **kwargs):
        super(RandlaConv, self).__init__(aggr='mean') #actual aggr in randlanet is sum, but mean is similar
        self.ratio = ratio
        self.k = k
        self.point_pos_nn = point_pos_nn
        self.attention_nn = attention_nn
        self.global_nn = global_nn

    def forward(self, data):
        x, pos, batch = data
        idx = torch.randint(0, pos.shape[0], (math.floor(pos.shape[0]*self.ratio),))
        row, col = knn(pos, pos[idx], self.k, batch, batch[idx])
        edge_index = torch.stack([row, col], dim=0)
        import pdb; pdb.set_trace()
        x = self.propagate(edge_index, pos=(pos, pos[idx]), x=x)

        pos, batch = pos[idx], batch[idx]
        data = (x, pos, batch)
        return data 

    def message(self, pos_i, pos_j, x_j):

        if x_j is None:
            x_j = pos_j

        vij = pos_i - pos_j

        dij = torch.norm(vij, dim=1).unsqueeze(1)

        relPointPos = torch.cat([
            pos_i,
            pos_j,
            vij,
            dij
        ])

        rij = self.point_pos_nn(relPointPos)

        fij_hat = torch.cat([x_j, rij])

        g_fij = self.attention_nn(fij_hat)

        s_ij = F.softmax(g_fij)

        msg = s_ij * fij_hat
        
        return msg

    def update(self, aggr_out):
        return self.global_nn(aggr_out)

# class RandlaDilatedResidual(torch.nn.Module):

#     def __init__(self, input_nn = None, output_nn = None, residual_nn = None, *args, **kwargs):
#         super(RandlaDilatedResidual, self).__init__()

#         self.input_nn = input_nn
#         self.output_nn = output_nn
#         self.residual_nn = residual_nn

#         self.conv = RandlaConv(**kwargs)

#     def forward(self, data):

class RandLANet(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(RandLANet, self).__init__()

        self.conv = RandlaConv(global_nn=kwargs['down_conv_nn'], **kwargs)

    def forward(self, data):
        return self.conv(data)
