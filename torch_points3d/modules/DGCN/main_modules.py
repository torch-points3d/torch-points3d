import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, LeakyReLU, Dropout
from torch_points3d.core.common_modules import Seq

from torch_points3d.applications.pointnet2 import PointNet2

from torch_points3d.modules.DGCN.modules import DeconvModule
from torch_points3d.core.common_modules.dense_modules import MLP1D, Conv1D

class PointGenerator(nn.Module):
    def __init__(self, 
        point_scales,
        latent_space,
        config, 
        # num_k=20, 
        # softmax=True, 
        # out_nn=[512, 256, 64, 3], 
        # in_feats=[], 
        # out_feats=[],
        # latent_space=128,
        #initial_features=4096,
    ):
        super(PointGenerator, self).__init__()
        self.point_scales = point_scales
        self.in_feats = config.in_feats
        self.out_feats = config.out_feats
        self.latent_space = latent_space
        self.out_nn = config.out_nn
        self.softmax = config.use_softmax
        # self.num_point = num_point
        self.num_k = config.knn

        assert len(self.point_scales) == len(self.in_feats) == len(self.out_feats) 

        initial_features = (point_scales[0] // 2) * self.in_feats[0]

        self.fc1 = nn.Sequential(
            nn.Linear(latent_space, initial_features),
            nn.BatchNorm1d(initial_features),  # 128,32
            nn.LeakyReLU()
        )

        self.interpolations = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(point_scales)):
            self.interpolations.append(DeconvModule(self.in_feats[i], self.out_feats[i], point_scales[i]//2, num_k=self.num_k, softmax=self.softmax))

            out_nn = self.out_nn[:]
            out_nn[0] += self.out_feats[i]

            mlp = nn.Sequential(
                MLP1D(out_nn, bn=True, bias=False),
                nn.Tanh(),
            )

            self.mlps.append(mlp)

        # self.mlp1 = nn.Sequential(
        #     nn.Conv1d(512+32, 256, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(256, 64, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(64, 3, 1),
        #     nn.Tanh()
        # )
        # self.mlp2 = nn.Sequential(
        #     nn.Conv1d(512+64, 256, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(256, 64, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(64, 3, 1),
        #     nn.Tanh()          
        # )
        # self.mlp3 = nn.Sequential(
        #     nn.Conv1d(512+128, 256, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(256, 64, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(64, 3, 1),
        #     nn.Tanh()
        # )
        # self.mlp4 = nn.Sequential(
        #     nn.Conv1d(512, 256, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(256, 64, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(64, 3, 1),
        #     nn.Tanh()
        # )

    def forward(self, x):
        #x = data.x

        batchsize = x.size()[0]
        x = self.fc1(x)
        x = x.view(batchsize, self.in_feats[0], self.point_scales[0]//2)  # Bx32x128
        xs = None

        out_points = []
        for i in range(len(self.point_scales)):
            x, g_x = self.interpolations[i](x, xs)
            xs = self.mlps[i](g_x) 
            out_points.append(xs)

        return out_points

        # x1, g_x1 = self.bilateral1(x)           # x1: Bx64x256
        # x1s = self.mlp1(g_x1)                   # Bx3x256
        # #print('x1: {} x1s: {}'.format(x1.size(), x1s.size()))

        # x2, g_x2 = self.bilateral2(x1, x1s)     # x2: Bx128x512
        # x2s = self.mlp2(g_x2)                   # Bx3x512
        # #print('x2: {} x2s: {}'.format(x2.size(), x2s.size()))
        
        # x3, g_x3 = self.bilateral3(x2, x2s)          # x3: Bx256x1024
        # x3s = self.mlp3(g_x3)                   # Bx3x1024
        # #print('x3: {} x3s: {}'.format(x3.size(), x3s.size()))
        
        # x4 = self.bilateral4(x3, x3s)                # x4: Bx512x2048
        # x4s = self.mlp4(x4)                     # Bx3x2048
        # #print('x4: {} x4s: {}'.format(x4.size(), x4s.size()))
        # #exit()
        return x1s, x2s, x3s, x4s

    def backward(self, loss):
        loss.backward()


class Discriminator(nn.Module):
    def __init__(self, fc_layers, classifier):
        #fc_layers = config.fc_layers
        #classifier = config.classifier

        super(Discriminator, self).__init__()
        #assert len(point_scales) == len(fc_layers)

        #self.fc_layers = nn.ModuleList()
        #for i in range(len(point_scales)):
        self.fc1 = Seq()
        for j in range(1, len(fc_layers)):
            self.fc1.append(Conv1D(fc_layers[j - 1], fc_layers[j], bn=True, bias=False, activation=nn.LeakyReLU(inplace=True)))
        self.fc1.append(Conv1D(fc_layers[-1], 1, activation=None, bias=True, bn=False)) #output a binary prediction (fake/real)
        #print(self.FC_layer)
        #    self.fc_layers.append(FC_layer)
        
        #self.classifiers = nn.ModuleList()
        #for i in range(len(point_scales)):
        self.classifier = self.build_backbone(classifier)

    def build_backbone(self, classifier):
        backbone_builder = globals().copy()[classifier]
        model = backbone_builder(
            architecture="encoder",
            input_nc=0,
            num_layers=3,
            #in_feat=self._model_opt.in_feat,
            #num_layers=self._model_opt.num_layers,
            #output_nc=self._num_classes,
            multiscale=True,
        )
        return model


    def forward(self, data):
        # data: array of points generated with different scales 

        # predictions = []
        # for i in range(len(point_scales)):
        data = self.classifier(data)
        last_feature = data.x
        # print('hi')
        # print(self.fc1)
        # print('hi2')
        #print(self.fc1.forward)
        out = self.fc1(last_feature).transpose(1, 2).contiguous().view((-1, 1))
        #    predictions.append(out)
        #print(out)
        return out

    def backward(self, loss):
        loss.backward()