from typing import Any
import torch
torch.backends.cudnn.enabled = False
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import knn_interpolate
from torch_geometric.nn import radius, global_max_pool
import etw_pytorch_utils as pt_utils
from .modules import *
from models.unet_base import UnetBasedModel, BaseModel


class SegmentationModel(BaseModel):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, option, model_type, dataset, modules):
        super(SegmentationModel, self).__init__(option)

        self.SA_modules = nn.ModuleList()
        use_xyz = True
        self.loss_names = ['loss_seg']
        self.weight_classes = dataset.weight_classes

        c_in = input_channels = dataset.feature_dimension
        self._num_classes = dataset.num_classes

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=use_xyz,
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=use_xyz,
            )
        )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
                use_xyz=use_xyz,
            )
        )
        c_out_2 = 256 + 256

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=16,
                radii=[0.4, 0.8],
                nsamples=[16, 32],
                mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
                use_xyz=use_xyz,
            )
        )
        c_out_3 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + input_channels, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))

        self.FC_layer = (
            pt_utils.Seq(128)
            .conv1d(128, bn=True)
            .dropout()
            .conv1d(self._num_classes, activation=None)
        )
        print(self)

    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
            Dimensions: [B, N, ...]
        """
        self.x = data.x.transpose(1, 2).contiguous()
        self.pos = data.pos.to("cuda", non_blocking=True)
        self.labels = torch.flatten(data.y).to("cuda", non_blocking=True)

    def forward(self):
        # type: (Pointnet2MSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        # torch.Size([32, 4096, 3]), torch.Size([32, 6, 4096])
        l_xyz, l_features = [self.pos], [self.x]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        self.output = self.FC_layer(l_features[0]).transpose(1, 2).contiguous().view((-1, self._num_classes))
        return self.output

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results

        self.loss_seg = F.cross_entropy(self.output, self.labels.long(),
                                        weight=self.weight_classes.to(self.output.device))
        self.loss_seg.backward()


# class SegmentationModel(UnetBasedModel):
#     def __init__(self, option, model_type, dataset, modules):
#         # call the initialization method of UnetBasedModel
#         UnetBasedModel.__init__(self, option, model_type, dataset, modules)

#         nn = option.mlp_cls.nn
#         self.dropout = option.mlp_cls.get('dropout')
#         self.lin1 = torch.nn.Linear(nn[0], nn[1])
#         self.lin2 = torch.nn.Linear(nn[1], nn[2])
#         self.lin3 = torch.nn.Linear(nn[2], dataset.num_classes)

#         self.loss_names = ['loss_seg']

#     def set_input(self, data):
#         """Unpack input data from the dataloader and perform necessary pre-processing steps.
#         Parameters:
#             input: a dictionary that contains the data itself and its metadata information.
#             Dimensions: [B, N, ...]
#         """
#         self.input = data
#         self.labels = torch.flatten(data.y)

#     def forward(self) -> Any:
#         """Standard forward"""
#         data = self.model(self.input)
#         x = data.x.squeeze(-1)
#         x = x.view((-1, x.shape[1]))
#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.lin2(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.lin3(x)
#         self.output = x
#         return self.output

#     def backward(self, debug=False):
#         """Calculate losses, gradients, and update network weights; called in every training iteration"""
#         # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
#         # calculate loss given the input and intermediate results

#         if debug:
#             print(self.output, torch.isnan(self.output).any(), torch.unique(self.labels))
#             print(self.output.shape, self.labels.shape)

#             try:
#                 self.loss_seg = F.cross_entropy(self.output, self.labels.long())
#                 if torch.isnan(self.loss_seg):
#                     import pdb
#                     pdb.set_trace()
#                 self.loss_seg.backward()
#             except:
#                 import pdb
#                 pdb.set_trace()
#             grad_ = self.model.down._local_nn[0].conv.weight.grad
#             print(torch.sum(grad_) != 0)
#         else:
#             self.loss_seg = F.cross_entropy(self.output, self.labels.long())
#             self.loss_seg.backward()
