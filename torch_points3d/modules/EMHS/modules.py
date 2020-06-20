from typing import *
import torch
from torch import nn


class EMHSDoubleLayer(nn.Module):
    def __init__(
        self,
        input_nc: int = None,
        output_nc: int = None,
        feat_dim: int = None,
        use_attention: bool = True,
        latent_classes: List = None,
        kernel_size: List = [3, 3, 3],
    ):

        super().__init__()

        self.elm_1 = EquivariantLinearMapsModule(input_nc, feat_dim, kernel_size)
        self.elm_2 = EquivariantLinearMapsModule(feat_dim, feat_dim, kernel_size)

    def forward(self, x, pos):
        pass


class EquivariantLinearMapsModule(nn.Module):
    def __init__(self, input_nc: int = 3, output_nc: int = 64, kernel_size: List = [3, 3, 3]):

        super().__init__()

        self.lin = nn.Linear(input_nc, output_nc)
        self.conv = nn.Conv3d(input_nc, output_nc, kernel_size=kernel_size)

    def forward(self, x, pos, pooling_idx, broadcasting_idx):

        return self.lin(torch.cat([x, pos], dim=-1))
