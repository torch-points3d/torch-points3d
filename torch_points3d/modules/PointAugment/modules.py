import torch

torch.autograd.set_detect_anomaly(True)
from torch import nn
from random import shuffle
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_points3d.core.common_modules.dense_modules import MLP, MLP1D, Conv1D
from torch_points3d.utils.model_utils import freeze_params, unfreeze_params
from torch_points3d.applications import models
from torch_points3d.core.common_modules.base_modules import Seq


class AugmentationModule(nn.Module):

    """
    PointAugment: an Auto-Augmentation Framework for Point Cloud Classification
    https://arxiv.org/pdf/2002.10876.pdf
    """

    def __init__(self, config, conv_type="DENSE"):
        super(AugmentationModule, self).__init__()
        self._conv_type = conv_type

        if conv_type == "DENSE":
            #per point feature extraction
            self.nn_raising = MLP1D(config.nn_raising)
            #shape-wise regression
            self.nn_rotation = MLP1D(config.nn_rotation)
            #point-wise regression
            self.nn_translation = MLP1D(config.nn_translation)
        else:
            self.nn_raising = MLP(config.nn_raising)
            self.nn_rotation = MLP(config.nn_rotation)
            self.nn_translation = MLP(config.nn_translation)

    def forward(self, data):
        if 3 < data.pos.dim() and data.pos.dim() <= 1:
            raise Exception("data.pos doesn t have the correct dimension. Should be either 2 or 3")
        
        if self._conv_type == "DENSE":
            batch_size = data.pos.shape[0]
            num_points = data.pos.shape[1]
            F = self.nn_raising(data.pos.permute(0, 2, 1))
            G, _ = F.max(-1)
            noise_rotation = torch.randn(G.size()).to(G.device)
            noise_translation = torch.randn(F.size()).to(F.device)

            feature_rotation = [noise_rotation, G]
            feature_translation = [F, G.unsqueeze(-1).repeat((1, 1, num_points)), noise_translation]

            features_rotation = torch.cat(feature_rotation, dim=1).unsqueeze(-1)
            features_translation = torch.cat(feature_translation, dim=1)

            M = self.nn_rotation(features_rotation).view((batch_size, 3, 3))
            D = self.nn_translation(features_translation).permute(0, 2, 1)

            new_data = data.clone()
            new_data.pos = D + new_data.pos @ M
        else:
            batch_size = data.pos.shape[0]
            num_points = data.pos.shape[1]
            F = self.nn_raising(data.pos.permute(0, 2, 1))
            G, _ = F.max(-1)
            noise_rotation = torch.randn(G.size()).to(G.device)
            noise_translation = torch.randn(F.size()).to(F.device)

            feature_rotation = [noise_rotation, G]
            feature_translation = [F, G.unsqueeze(-1).repeat((1, 1, num_points)), noise_translation]

            features_rotation = torch.cat(feature_rotation, dim=1).unsqueeze(-1)
            features_translation = torch.cat(feature_translation, dim=1)

            M = self.nn_rotation(features_rotation).view((batch_size, 3, 3))
            D = self.nn_translation(features_translation).permute(0, 2, 1)

            new_data = data.clone()
            new_data.pos = D + new_data.pos @ M

        return new_data


class ClassifierModule(nn.Module):
    def __init__(self, model_opt, input_nc, num_classes):
        super(ClassifierModule, self).__init__()

        self._input_nc = input_nc
        self._num_classes = num_classes
        self._model_opt = model_opt

        backbone_option = model_opt.backbone
        backbone_cls = getattr(models, backbone_option.model_type)
        self.backbone_model = backbone_cls(architecture="encoder", input_nc=input_nc, config=backbone_option)

        # Last MLP
        last_mlp_opt = model_opt.mlp_cls

        self.FC_layer = Seq()
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.append(Conv1D(last_mlp_opt.nn[i - 1], last_mlp_opt.nn[i], bn=True, bias=False))
        if last_mlp_opt.dropout:
            self.FC_layer.append(torch.nn.Dropout(p=last_mlp_opt.dropout))

        self.FC_layer.append(Conv1D(last_mlp_opt.nn[-1], self._num_classes, activation=None, bias=True, bn=False))


    def forward(self, data):
        # returns `y` (class labels predicted by the fully connected layers) and `F_g` (per-shape global features)
        if self._model_opt.model_type == "KPConv":  # KPConv needs added ones for its x features
            data = AddOnes()(data)
            data.x = (
                torch.cat([data.x, data.ones.to(data.pos.device)], dim=-1)
                if data.x is not None
                else data.ones.to(data.pos.device)
            )
        data = self.backbone_model(data)
        last_feature = data.x

        self.output = self.FC_layer(last_feature).transpose(1, 2).contiguous().view((-1, self._num_classes))

        return F.log_softmax(self.output.squeeze(), dim=-1), last_feature
