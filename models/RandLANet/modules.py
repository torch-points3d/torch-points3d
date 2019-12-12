
import torch 
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, knn
from models.core_modules import *
import math

class RandlaConv(MessagePassing):
    '''
        Implements both the Local Spatial Encoding and Attentive Pooling blocks from 
        RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds
        https://arxiv.org/pdf/1911.11236

    '''

    def __init__(self, ratio=None, k=None, point_pos_nn=None, attention_nn=None, global_nn=None, **kwargs):
        super(RandlaConv, self).__init__(aggr='mean') #actual aggr in randlanet is sum, but mean is similar
        print("initing randla conv", locals())
        self.ratio = ratio
        self.k = k
        self.point_pos_nn = MLP(point_pos_nn)
        self.attention_nn = MLP(attention_nn)
        self.global_nn = MLP(global_nn)

    def forward(self, data):
        x, pos, batch = data
        idx = torch.randint(0, pos.shape[0], (math.floor(pos.shape[0]*self.ratio),))
        row, col = knn(pos, pos[idx], self.k, batch, batch[idx])
        edge_index = torch.stack([col, row], dim=0)
        x = self.propagate(edge_index, x=x, pos=(pos, pos[idx]))

        pos, batch = pos[idx], batch[idx]
        data = (x, pos, batch)
        return data 

    def message(self, x_j, pos_i, pos_j):

        if x_j is None:
            x_j = pos_j

        vij = pos_i - pos_j

        dij = torch.norm(vij, dim=1).unsqueeze(1)

        relPointPos = torch.cat([
            pos_i,
            pos_j,
            vij,
            dij
        ], dim=1)

        rij = self.point_pos_nn(relPointPos)

        fij_hat = torch.cat([x_j, rij], dim=1)

        g_fij = self.attention_nn(fij_hat)

        s_ij = F.softmax(g_fij)

        msg = s_ij * fij_hat
        
        return msg

    def update(self, aggr_out):
        return self.global_nn(aggr_out)

#This is not the real randla-net - it is basically pointnet++ using the local spatial encoding 
#and attentative pooling blocks from randla-net as the convolution. 
class RandLANet(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(RandLANet, self).__init__()

        self.conv = RandlaConv(global_nn=kwargs['down_conv_nn'], **kwargs)

    def forward(self, data):
        return self.conv(data)
