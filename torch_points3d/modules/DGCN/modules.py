import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_points_kernels as tp

from torch_points3d.core.base_conv.dense import *
from torch_points3d.core.common_modules.dense_modules import *
from torch_points3d.utils.model_building_utils.activation_resolver import get_activation
from torch_points3d.core.common_modules.base_modules import BaseModule
from torch_points3d.modules.DGCN.helpers import get_edge_features_xyz, get_edge_features


class DeconvModule(BaseModule):
    def __init__(
        self,
        Fin,
        Fout,
        maxpool,
        num_k,
        softmax,
        **kwargs
    ):
        super(DeconvModule, self).__init__()

        self.nn = nn.ModuleDict()

        self.nn["maxpool"] = nn.MaxPool2d((1,maxpool),(1,1))
        self.nn["upsample_cov"] = BilateralInterpolationModule(Fin, Fout, num_k//2, 1, softmax=softmax)
        self.nn["bn_uc"] = nn.BatchNorm1d(Fout)
        self.nn["relu_uc"] = nn.LeakyReLU(inplace=True)

        # self.maxpool = 
        
        # self.upsample_cov = BilateralInterpolationModule(Fin, Fout, num_k//2, 1, softmax=softmax)   #(256->512)
        # self.bn_uc = nn.BatchNorm1d(Fout)
        # self.relu_uc = nn.LeakyReLU(inplace=True)        
        
        self.nn["fc"] = nn.Sequential(
            nn.Linear(Fin, Fin),
            nn.BatchNorm1d(Fin),
            nn.LeakyReLU(inplace=True),
            nn.Linear(Fin, Fout),
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True),
        )
        self.nn["g_fc"] = nn.Sequential(
            nn.Linear(Fout,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, pc):
        """
        Parameters
        ----------
        data: Data
            x -- Previous features [B, C, N]
            pos -- Previous positions [B, N, 3]
        """
        # x = data.x
        # pc = data.pos.permute(0,2,1)

        batchsize, _, point_num = x.size()
        xs = self.nn["maxpool"](x)
        xs = xs.view(batchsize,-1)
        xs = self.nn["fc"](xs)
        
        g = self.nn["g_fc"](xs)
        g = g.view(batchsize, -1, 1)
        g = g.repeat(1, 1, 2*point_num)

        xs = xs.view(batchsize,-1,1)
        xs = xs.repeat(1, 1, 2*point_num)

        x_ec = self.nn["relu_uc"](self.nn["bn_uc"](self.nn["upsample_cov"](x, pc)))
        x_out = torch.cat((xs, x_ec), 1)

        g_out = torch.cat((g, x_ec), dim=1)
        
        return x_out, g_out

    def __repr__(self):
        return "{}: {} ({})".format(self.__class__.__name__, self.nb_params, self.nn)

class BilateralInterpolationModule(BaseModule):
    """ Edge Convolution using 1x1 Conv h
    [B, Fin, N] -> [B, Fout, N]
    """
    def __init__(self, Fin, Fout, k, num, softmax=True):
        super(BilateralInterpolationModule, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.softmax = softmax
        self.num = num

        # self.conv = conv2dbr(2*Fin, Fout, [1, 20], [1, 20])
        #self.conv1 = conv2dbr(2*Fin, 2*Fin, 1 ,1)
        #self.conv2 = Conv2D(2*Fin, 2*Fout, [1, 2*k], [1, 1])
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*Fin, 2*Fout,  [1, 2*k],  [1, 1]),
            nn.BatchNorm2d(2*Fout),
            nn.ReLU(True),
        )

        self.conv_xyz = nn.Sequential(
            nn.Conv2d(6, 16, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_fea = nn.Sequential(
            nn.Conv2d(2*Fin, 16, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_all = nn.Sequential(
            nn.Conv2d(16, 64, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 2*Fin, 1),
            nn.BatchNorm2d(2*Fin),
            nn.LeakyReLU(inplace=True)
        )

        self.inte_conv_hk = nn.Sequential(
            #nn.Conv2d(2*Fin, 4*Fin, [1, k//2], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.Conv2d(2*Fin, 4*Fin, [1, k//2+1], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.BatchNorm2d(4*Fin),
            nn.LeakyReLU(inplace = True)
        )

    def forward(self, x, pc):
        B, Fin, N = x.size()
        has_points = pc is not None and pc.shape[-1] > 0

        #x = get_edge_features(x, self.k, self.num); # [B, 2Fin, N, k]
        if has_points:
            x, y = get_edge_features_xyz(x, pc, self.k, self.num); # feature x: [B, 2Fin, N, k] coordinate y: [B, 6, N, k]
        else:
            x = get_edge_features(x, self.k, self.num); # feature x: [B, 2Fin, N, k] coordinate y: [B, 6, N, k]

        if has_points:
            w_fea = self.conv_fea(x)
            w_xyz = self.conv_xyz(y)
            w = w_fea * w_xyz
            w = self.conv_all(w)
            if self.softmax == True:
                w = F.softmax(w, dim=-1)    # [B, Fout, N, k] -> [B, Fout, N, k]
        
        # -------------learn_v2----------------------
        BB, CC, NN, KK = x.size()
        #x = self.conv1(x)
        inte_x = self.inte_conv_hk(x)                                   # Bx2CxNxk/2
        inte_x = inte_x.transpose(2, 1)                                 # BxNx2Cxk/2
        inte_x = inte_x.contiguous().view(BB, NN, CC, 2, KK//2)       # BxNxCx2x(k//2+1)
        inte_x = inte_x.contiguous().view(BB, NN, CC, KK)             # BxNxCx(k+2)
      
        inte_x = inte_x.permute(0, 2, 1, 3)                             # BxCxNx(k+2)
        if has_points:
            inte_x = inte_x * w
        
        # Here we concatenate the interpolated feature with the original feature.
        merge_x = torch.cat((x, inte_x), 3)                             # BxCxNx2k
        
        # Since conv2 uses a wide kernel size, the process of sorting by distance can be omitted.
        x = self.conv2(merge_x) # [B, 2*Fout, N, 1]

        x = x.unsqueeze(3)                    # BxkcxN
        x = x.contiguous().view(B, self.Fout, 2, N)
        x = x.contiguous().view(B, self.Fout, 2*N)

        assert x.shape == (B, self.Fout, 2*N)
        return x