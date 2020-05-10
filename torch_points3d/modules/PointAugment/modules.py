import torch
from torch import nn
from random import shuffle
import torch.nn.functional as F
from torch_points3d.core.common_modules.dense_modules import MLP1D
from torch_points3d.utils.model_utils import freeze_params, unfreeze_params


class PointAugment(nn.Module):

    """
    PointAugment: an Auto-Augmentation Framework for Point Cloud Classification
    https://arxiv.org/pdf/2002.10876.pdf
    """

    def __init__(self, input_model, num_features, num_classes, conv_type, opt):
        super(PointAugment, self).__init__()
        self.input_model = input_model
        self._num_features = num_features
        self._conv_type = conv_type.lower()
        self._num_classes = num_classes
        self._shuffle = opt.shuffle
        self._opt = opt.to_container()
        self.model = nn.ModuleDict()
        self._build()

    def _build(self):
        if self._conv_type == "dense":
            nn_raising = [3] + self._opt["nn_raising"]
            nn_rotation = [nn_raising[-1] * 2] + self._opt["nn_rotation"] + [9]
            nn_translation = [nn_raising[-1] * 3] + self._opt["nn_translation"] + [3]
            self.model["nn_raising"] = MLP1D(nn_raising)
            self.model["nn_rotation"] = MLP1D(nn_rotation)
            self.model["nn_translation"] = MLP1D(nn_translation)

    def _forward(self, data):
        pos = data.pos

        if pos.dim() == 3:
            batch_size = pos.shape[0]
            num_points = pos.shape[1]
            F = self.model["nn_raising"](pos.permute(0, 2, 1))
            G = F.mean(-1)
            noise_rotation = torch.randn(G.size()).to(G.device)
            noise_translation = torch.randn(F.size()).to(F.device)

            feature_rotation = [noise_rotation, G]
            feature_translation = [F, G.unsqueeze(-1).repeat((1, 1, num_points)), noise_translation]

            if self._shuffle:
                shuffle(feature_rotation)
                shuffle(feature_translation)

            features_rotation = torch.cat(feature_rotation, dim=1).unsqueeze(-1)
            features_translation = torch.cat(feature_translation, dim=1)

            M = self.model["nn_rotation"](features_rotation).view((batch_size, 3, 3))
            T = self.model["nn_translation"](features_translation).permute(0, 2, 1)

            new_data = data.clone()
            new_data.pos = T + new_data.pos @ M

        return new_data

    def set_input(self, data):
        self._data = data
        self.labels = self._data.y.flatten()

    def forward(self, data):
        self.set_input(data)

        data_augmented = self._forward(self._data)

        self.output, feat = self.input_model(self._data)
        output_augmented, feat_augmented = self.input_model(data_augmented)

        self.loss_p = F.nll_loss(self.output, self.labels)
        self.loss_pa = F.nll_loss(output_augmented, self.labels)

        pv = max(1, torch.exp(torch.sum(self.output.exp() * F.one_hot(self.labels, self._num_classes))))
        self.loss_aug = torch.abs(1 - torch.exp(self.loss_pa - pv * self.loss_p))
        self.loss_reg = F.mse_loss(feat, feat_augmented)
        self.loss = self.loss_pa + self.loss_p + self.loss_reg
        return self.output

    def backward(self):
        # Optimize augmentor using LA = |1.0 − exp[L(P′) − ρL(P)]|.
        freeze_params(self.input_model)
        unfreeze_params(self.model)
        self.loss_aug.backward(retain_graph=True)

        # Optimize classifier
        freeze_params(self.model)
        unfreeze_params(self.input_model)
        self.loss.backward()
