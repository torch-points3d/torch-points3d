import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, LeakyReLU, Dropout
from torch_points3d.core.common_modules import Seq

from torch_points3d.applications.pointnet2 import PointNet2

from torch_points3d.modules.DGCN.modules import DeconvModule
from torch_points3d.core.common_modules.dense_modules import MLP1D, Conv1D

class PointGenerator(nn.Module):
    def __init__(self, 
        scales,
        latent_space,
        config,
    ):
        super(PointGenerator, self).__init__()
        self.scales = scales
        self.in_feats = config.in_feats
        self.out_feats = config.out_feats
        self.latent_space = latent_space
        self.out_nn = config.out_nn
        self.softmax = config.use_softmax
        self.knn = config.knn

        #assert len(self.scales) == len(self.in_feats) == len(self.out_feats) 

        initial_features = (self.scales[0] // 2) * self.in_feats[0]

        self.fc1 = nn.Sequential(
            nn.Linear(self.latent_space, initial_features),
            nn.BatchNorm1d(initial_features),  # 128,32
            nn.LeakyReLU()
        )

        self.interpolations = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(self.scales)):
            self.interpolations.append(DeconvModule(self.in_feats[i], self.out_feats[i], self.scales[i]//2, num_k=self.knn, softmax=self.softmax))

            out_nn = self.out_nn[:]
            out_nn[0] += self.out_feats[i]

            mlp = nn.Sequential(
                MLP1D(out_nn, bn=True, bias=False),
                nn.Tanh(),
            )

            self.mlps.append(mlp)

    def forward(self, x):

        batchsize = x.size()[0]
        x = self.fc1(x)
        x = x.view(batchsize, self.in_feats[0], self.scales[0]//2)  # Bx32x128
        xs = None

        out_points = []
        for i in range(len(self.scales)):
            x, g_x = self.interpolations[i](x, xs)
            xs = self.mlps[i](g_x) 
            out_points.append(xs)

        return out_points

    def backward(self, loss):
        loss.backward()


class Discriminator(nn.Module):
    def __init__(self, fc_layers, classifier):
        super(Discriminator, self).__init__()
        
        self.fc1 = Seq()
        for j in range(1, len(fc_layers)):
            self.fc1.append(Conv1D(fc_layers[j - 1], fc_layers[j], bn=True, bias=False, activation=nn.LeakyReLU(inplace=True)))
        self.fc1.append(Conv1D(fc_layers[-1], 1, activation=None, bias=True, bn=False)) #output a binary prediction (fake/real)

        self.classifier = self.build_backbone(classifier)

    def build_backbone(self, classifier):
        backbone_builder = globals().copy()[classifier]
        model = backbone_builder(
            architecture="encoder",
            input_nc=0,
            num_layers=3,
            multiscale=True,
        )
        return model


    def forward(self, data):
        data = self.classifier(data)
        last_feature = data.x
        out = self.fc1(last_feature).transpose(1, 2).contiguous().view((-1, 1))
        return out

    def backward(self, loss):
        loss.backward()