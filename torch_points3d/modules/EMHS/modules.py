from typing import *
from omegaconf.listconfig import ListConfig
import numpy as np
from torch_scatter import scatter_mean
import torch
from torch import nn
from torch.nn import functional as F
from torch_points3d.utils.enums import AttentionType


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
        attention_type=AttentionType.CLL.value,
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
                    attention_type,
                )
            )

        self.model = nn.Sequential(*modules)

        if input_nc < output_nc:
            self.lin_first = nn.Linear(input_nc, feat_dim)
            self.norm_first = nn.BatchNorm1d(feat_dim)

        if output_nc < input_nc:
            self.lin_fast = nn.Linear(feat_dim, output_nc)
            self.norm_last = nn.BatchNorm1d(output_nc)

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
            return x + F.leaky_relu(self.norm_first(self.lin_first(xs[0])), negative_slope=0.2)
        else:
            return x + F.leaky_relu(self.norm_last(self.lin_fast(xs[0])), negative_slope=0.2)


class EquivariantLinearMapsModule(nn.Module):
    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 64,
        use_attention: bool = True,
        latent_dim: int = 50,
        kernel_size=[3, 3, 3],
        voxelization=[9, 9, 9],
        attention_type=AttentionType.CLL.value,
    ):

        super().__init__()

        assert attention_type in [v.value for v in AttentionType], "attention_type should be in {}".format(
            [v.value for v in AttentionType]
        )

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.use_attention = use_attention
        self.latent_dim = latent_dim
        self._voxelization = voxelization.to_container() if isinstance(voxelization, ListConfig) else voxelization

        self.lin = nn.Linear(input_nc, output_nc)
        if self.use_attention:
            self.lin_attention = nn.Linear(input_nc, latent_dim)
            if attention_type == AttentionType.CLL.value:
                self.weight = torch.nn.Parameter(torch.zeros(output_nc, latent_dim, latent_dim))
                self._attention_ops = self._attention_cll_ops
            else:
                self.weight = torch.nn.Parameter(torch.zeros(output_nc, latent_dim))
                self._attention_ops = self._attention_cl_ops

        self.conv = nn.Conv3d(input_nc, output_nc, kernel_size=kernel_size, padding=1)
        self.norm = nn.BatchNorm1d(output_nc)

        if self.use_attention:
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def _attention_cll_ops(self, x):
        H = self.lin_attention(x)
        H = F.softmax(H, dim=-1)
        # (L, N), (N, C) -> (L, C)
        HX = torch.matmul(H.t(), x)  # size L x C: how much class l contains of feature c
        # (C', L, L), (L, C) -> (C', L, C)
        W4HX = torch.matmul(self.weight, HX)  # size Cp x L x C : how much channel c' for class l from channel c
        HW4HX = torch.einsum(
            "nl, mlc-> nmc", H, W4HX
        )  # size N x C'x C : for point n how much channel c' from channel c
        SHW4HX = torch.sum(HW4HX, dim=-1)  # size N x C' : for point n how much channel c'
        return SHW4HX

    def _attention_cl_ops(self, x):
        H = self.lin_attention(x)
        H = F.softmax(H, dim=-1)
        # (L, N), (N, C) -> (L, C)
        HX = torch.matmul(H.t(), x)  # size L x C: how much class l contains of feature c
        W5HX = (
            self.weight[:, :, None] * HX[None, :, :]
        )  # size Cp x L x C : how much channel c' for class l from channel c
        HW5HX = torch.einsum(
            "nl, mlc-> nmc", H, W5HX
        )  # size N x C'x C : for point n how much channel c' from channel c
        SHW5HX = torch.sum(HW5HX, dim=-1)  # size N x C' : for point n how much channel c'
        return SHW5HX

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
        return F.leaky_relu(self.norm(inner_equivariant_map + outer_equivariant_map), negative_slope=0.2)
