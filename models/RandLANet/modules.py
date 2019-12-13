
import torch 
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, knn
from models.core_modules import *
import math

class RandlaConv(MessagePassing, BaseKNNConvolution):
    '''
        Implements both the Local Spatial Encoding and Attentive Pooling blocks from 
        RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds
        https://arxiv.org/pdf/1911.11236

    '''

    def __init__(self, ratio=None, k=None, point_pos_nn=None, attention_nn=None, global_nn=None, **kwargs):
        MessagePassing.__init__(self, aggr='add')
        BaseKNNConvolution.__init__(self, ratio=ratio, k=k, sampling_strategy='random') 
        #torch.nn.Module.__init__ will be called twice, but this should be fine

        self.point_pos_nn = MLP(point_pos_nn)
        self.attention_nn = MLP(attention_nn)
        self.global_nn = MLP(global_nn)

    def conv(self, x, pos, edge_index):
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

        s_ij = F.softmax(g_fij)

        msg = s_ij * fij_hat
        
        return msg

    def update(self, aggr_out):
        return self.global_nn(aggr_out)

class DialatedResidualBlock(BaseResnetBlock):

    def __init__(self, indim, outdim, ratio1, ratio2, point_pos_nn1, point_pos_nn2, 
            attention_nn1, attention_nn2, global_nn1, global_nn2, *args, **kwargs):

        super(DialatedResidualBlock, self).__init__(indim, outdim, outdim)

        self.conv1 = RandlaConv(ratio1, 16, point_pos_nn1, attention_nn1, global_nn1)
        self.conv2 = RandlaConv(ratio2, 16, point_pos_nn2, attention_nn2, global_nn2)

    def convolution(self, data):
        *data, idx1 = self.conv1(data) #calls the forward function of BaseKNNConvolution
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

        self.conv = DialatedResidualBlock(
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
        return self.conv(data)

#This is not the real randla-net - it is basically pointnet++ using the local spatial encoding 
#and attentative pooling blocks from randla-net as the convolution. 
# class RandLANet(torch.nn.Module):

#     def __init__(self, *args, **kwargs):
#         super(RandLANet, self).__init__()

#         self.conv = RandlaConv(global_nn=kwargs['down_conv_nn'], **kwargs)

#     def forward(self, data):
#         return self.conv(data)
