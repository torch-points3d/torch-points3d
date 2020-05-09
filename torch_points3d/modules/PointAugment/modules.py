import torch
from torch import nn
from random import shuffle
from torch_points3d.core.common_modules.dense_modules import MLP1D


class PointAugment(nn.Module):

    """
    PointAugment: an Auto-Augmentation Framework for Point Cloud Classification
    https://arxiv.org/pdf/2002.10876.pdf
    """

    def __init__(self, num_features, conv_type, opt):
        super(PointAugment, self).__init__()
        self._num_features = num_features
        self._conv_type = conv_type.lower()
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

    def forward(self, data):
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
