from typing import *
import numpy as np
import torch
from torch import nn
from torch_points3d.modules.EMHS.modules import EMHSLayer


class Inputs(NamedTuple):
    pos: torch.Tensor
    x: torch.Tensor
    pooling_idx: torch.Tensor
    broadcasting_idx: torch.Tensor


class EMHSModel(nn.Module):
    def __init__(
        self,
        input_nc: int = None,
        output_nc: int = None,
        num_layers: int = 56,
        module_name: List = None,
        num_elm: int = 2,
        use_attention: bool = True,
        layers_slice: List = None,
        latent_classes: List = None,
        voxelization: List = [9, 9, 9],
        kernel_size: List = [3, 3, 3],
        feat_dim: int = 64,
    ):

        assert input_nc is not None, "input_nc is undefined"
        assert output_nc is not None, "output_nc is undefined"
        assert module_name is not None, "module_name is undefined"
        assert layers_slice is not None, "layers_slice is undefined"
        if use_attention:
            assert latent_classes is not None, "latent_classes is undefined"
            assert len(latent_classes) == len(layers_slice), "latent_classes and layers_slice should have the same size"

        # VALIDATION FOR IDX SLICES
        layers_idx = []
        for ls in layers_slice:
            s, e = ls.split("-")
            for layer_idx in range(int(s), int(e)):
                layers_idx.append(layer_idx)
        assert len(np.unique(layers_idx)) == num_layers, (len(np.unique(layers_idx)), num_layers)

        super().__init__()

        for idx_ls, ls in enumerate(layers_slice):
            s, e = ls.split("-")
            for layer_idx in range(int(s), int(e)):
                is_first = layer_idx == min(layers_idx)
                is_last = layer_idx == max(layers_idx)
                module = EMHSLayer(
                    input_nc if is_first else feat_dim,
                    output_nc if is_last else feat_dim,
                    feat_dim,
                    num_elm,
                    use_attention if not is_last else False,
                    latent_classes[idx_ls] if latent_classes is not None else None,
                    kernel_size,
                )
                self.add_module(str(layer_idx), module)

        print(self)

    def forward(self, x, consecutive_cluster, cluster_non_consecutive):
        for m in self._modules.values():
            x = m(x, consecutive_cluster, cluster_non_consecutive)
        return x
