
import torch 
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, knn
from models.core_modules import *
from models.core_sampling_and_search import *
import math

class RandlaKernel(MessagePassing):
    '''
        Implements both the Local Spatial Encoding and Attentive Pooling blocks from 
        RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds
        https://arxiv.org/pdf/1911.11236

    '''

    def __init__(self, point_pos_nn=None, attention_nn=None, global_nn=None, *args, **kwargs):
        MessagePassing.__init__(self, aggr='add')

        self.point_pos_nn = MLP(point_pos_nn)
        self.attention_nn = MLP(attention_nn)
        self.global_nn = MLP(global_nn)

    def forward(self, x, pos, edge_index):
        x = self.propagate(edge_index, x=x, pos=pos)
        return x

    def message(self, x_j, pos_i, pos_j):

        if x_j is None:
            x_j = pos_j

        #compute relative position encoding 
        vij = pos_i - pos_j

        dij = torch.norm(vij, dim=1).unsqueeze(1)

        relPointPos = torch.cat([
            pos_i,
            pos_j,
            vij,
            dij
        ], dim=1)

        rij = self.point_pos_nn(relPointPos)

        #concatenate position encoding with feature vector
        fij_hat = torch.cat([x_j, rij], dim=1)

        #attentative pooling
        g_fij = self.attention_nn(fij_hat)

        s_ij = F.softmax(g_fij, -1)

        msg = s_ij * fij_hat
        
        return msg

    def update(self, aggr_out):
        return self.global_nn(aggr_out)

class RandlaConv(BaseConvolution):

    def __init__(self, ratio = None, k = None, *args, **kwargs):
        super(RandlaConv, self).__init__(RandomSampler(ratio), KNNNeighbourFinder(k), *args, **kwargs)

        self._conv = RandlaKernel(*args, **kwargs)

    @property
    def conv(self):
        return self._conv

    def forward(self, data):
        x, pos, batch = data
        idx = self.sampler(pos, batch)
        row, col = self.neighbour_finder(pos, pos[idx], batch, batch[idx])
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        data = (x, pos, batch, idx)
        return data

class DilatedResidualBlock(BaseResnetBlock):

    def __init__(self, indim, outdim, ratio1, ratio2, point_pos_nn1, point_pos_nn2, 
            attention_nn1, attention_nn2, global_nn1, global_nn2, *args, **kwargs):

        super(DilatedResidualBlock, self).__init__(indim, outdim, outdim)

        self.conv1 = RandlaConv(ratio1, 16, point_pos_nn1, attention_nn1, global_nn1)
        self.conv2 = RandlaConv(ratio2, 16, point_pos_nn2, attention_nn2, global_nn2)

    def conv(self, data):
        *data, idx1 = self.conv1(data)
        *data, idx2 = self.conv2(data)
        if idx1 is None:
            if idx2 is None:
                return (*data, None)
            else:
                return (*data, idx2)
        else:
            if idx2 is None:
                return (*data, idx1)
            else:
                return (*data, idx1[idx2])

class RandLANetRes(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        print('Init randlanetres with kwargs: ', kwargs)
        super(RandLANetRes, self).__init__()

        self._conv = DilatedResidualBlock(
            kwargs['indim'],
            kwargs['outdim'],
            kwargs['ratio'][0],
            kwargs['ratio'][1],
            kwargs['point_pos_nn'][0],
            kwargs['point_pos_nn'][1],
            kwargs['attention_nn'][0],
            kwargs['attention_nn'][1],
            kwargs['down_conv_nn'][0],
            kwargs['down_conv_nn'][1]
        )

    def forward(self, data):
        return self._conv(data)

class RandLANet(BaseConvolution):

    def __init__(self, *args, **kwargs):
        super(RandLANet, self).__init__()

        self.conv = RandlaConv(global_nn=kwargs['down_conv_nn'], **kwargs)

    def forward(self, data):
        return self.conv(data)
