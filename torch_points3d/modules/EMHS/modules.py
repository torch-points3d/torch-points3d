from typing import *
from omegaconf.listconfig import ListConfig
import numpy as np
from torch_scatter import scatter_mean
import torch
from torch import nn
from torch.nn import functional as F


class EMHSLayer(nn.Module):
    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 20,
        feat_dim: int = 64,
        num_elm: int = 2,
        use_attention: bool = True,
        latent_dim: int = 5,
        kernel_size: List[int] = [3, 3, 3],
        voxelization: List[int] = [9, 9, 9],
    ):

        assert num_elm > 0, "num_elm should be greater than 0"
        super().__init__()

        self._input_nc = input_nc
        self._output_nc = output_nc

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
                    voxelization,
                )
            )

        self.model = nn.Sequential(*modules)
        self.lin_first = nn.Linear(input_nc, feat_dim)
        self.lin_fast = nn.Linear(feat_dim, output_nc)

    def forward(
        self,
        x,
        consecutive_cluster,
        cluster_non_consecutive,
        unique_cluster_non_consecutive,
        batch=None,
        batch_size=None,
    ):
        xs = [x]
        for m in self.model._modules.values():
            x = m(
                x,
                consecutive_cluster,
                cluster_non_consecutive,
                unique_cluster_non_consecutive,
                batch=batch,
                batch_size=batch_size,
            )
            xs.append(x)
        if xs[0].shape[-1] == xs[-1].shape[-1]:
            return xs[0] + xs[-1]
        elif xs[0].shape[-1] < xs[-1].shape[-1]:
            return x + self.lin_first(xs[0])
        else:
            return x + self.lin_fast(xs[0])


class EquivariantLinearMapsModule(nn.Module):
    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 64,
        use_attention: bool = True,
        latent_dim: int = 50,
        kernel_size=[3, 3, 3],
        voxelization=[9, 9, 9],
    ):

        super().__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.use_attention = False  # use_attention
        self._voxelization = voxelization.to_container() if isinstance(voxelization, ListConfig) else voxelization

        if self.use_attention:
            self.lin = nn.Linear(input_nc, latent_dim)
            self.weight = torch.nn.Parameter(torch.zeros(latent_dim, latent_dim, input_nc, output_nc))
        else:
            self.lin = nn.Linear(input_nc, output_nc)
        self.conv = nn.Conv3d(input_nc, output_nc, kernel_size=kernel_size, padding=1)

    def _attention_ops(self, x):
        pi = self.lin(x)
        pi = F.softmax(pi, dim=-1)
        pi_x = torch.einsum("bpc, bpl -> bpc", x, pi)
        pi_t_w4 = torch.einsum("bpl, lmcd -> bpc", pi, self.weight)
        return torch.einsum("bpc, bpc -> bpc", pi_t_w4, pi_x)

    def forward(
        self,
        x,
        consecutive_cluster,
        cluster_non_consecutive,
        unique_cluster_non_consecutive,
        batch=None,
        batch_size=None,
    ):
        inner_equivariant_map = self.lin(x)
        if self.use_attention:
            inner_equivariant_map += self._attention_ops(x)

        if batch is None:
            grid = torch.zeros([self.input_nc] + self._voxelization).view((self.input_nc, -1))
            grid[:, torch.unique(cluster_non_consecutive)] = scatter_mean(x, consecutive_cluster, dim=0).t()
            grid = self.conv(grid.view(([1] + [self.input_nc] + self._voxelization)))
            outer_equivariant_map = grid.view((self.output_nc, -1))[:, cluster_non_consecutive].t()
        else:
            grid = torch.zeros((self.input_nc, np.product(self._voxelization) * batch_size))
            grid[:, unique_cluster_non_consecutive] = scatter_mean(x, consecutive_cluster, dim=0).t()
            grid = self.conv(grid.view(([batch_size] + [self.input_nc] + self._voxelization)))
            outer_equivariant_map = grid.view((self.output_nc, -1))[:, cluster_non_consecutive].t()
        return inner_equivariant_map + outer_equivariant_map
