from typing import *
import torch
from torch import nn
from torch.nn import functional as F


class EMHSLayer(nn.Module):
    def __init__(
        self,
        input_nc: int = None,
        output_nc: int = None,
        feat_dim: int = None,
        num_elm: int = 2,
        use_attention: bool = True,
        latent_dim: int = None,
        kernel_size: List = [3, 3, 3],
    ):

        assert num_elm > 0, "num_elm should be greater than 0"
        super().__init__()

        modules = []
        for idx_layer in range(num_elm):
            is_first = idx_layer == 0
            is_last = idx_layer == num_elm - 1
            modules.append(
                EquivariantLinearMapsModule(
                    input_nc if is_first else feat_dim,
                    output_nc if is_last else feat_dim,
                    use_attention,
                    latent_dim,
                    kernel_size,
                )
            )

        self.model = nn.Sequential(*modules)

    def forward(self, x, pos):
        pass


class EquivariantLinearMapsModule(nn.Module):
    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 64,
        use_attention: bool = True,
        latent_dim: int = 50,
        kernel_size: List = [3, 3, 3],
    ):

        super().__init__()

        self.use_attention = use_attention

        if self.use_attention:
            self.lin = nn.Linear(input_nc, latent_dim)
            self.weight = torch.nn.Parameter(torch.zeros(latent_dim, latent_dim, input_nc, output_nc))
        else:
            self.lin = nn.Linear(input_nc, output_nc)
        self.conv = nn.Conv3d(input_nc, output_nc, kernel_size=kernel_size)

    def _attention_ops(self, x):
        pi = self.lin(x)
        pi = F.softmax(pi, dim=-1)
        pi_x = torch.einsum("bpc, bpl -> bpc", x, pi)
        pi_t_w4 = torch.einsum("bpl, lmcd -> bpc", pi, self.weight)
        return torch.einsum("bpc, bpc -> bpc", pi_t_w4, pi_x)

    def forward(self, x, pooling_idx, broadcasting_idx):

        if self.use_attention:
            inner_equivariant_map = self._attention_ops(x)
        else:
            inner_equivariant_map = self.lin(x)

        outer_equivariant_map = 0

        return (inner_equivariant_map + outer_equivariant_map, pooling_idx, broadcasting_idx)
