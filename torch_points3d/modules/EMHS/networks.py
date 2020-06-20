from typing import *
import numpy as np
from torch import nn

from torch_points3d.modules.EMHS.modules import EMHSResidualLayer


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

        # VALIDATION FOR IDX SLICES
        layers_idx = []
        for ls in layers_slice:
            s, e = ls.split("-")
            for layer_idx in range(int(s), int(e)):
                layers_idx.append(layer_idx)
        assert len(np.unique(layers_idx)) == num_layers, (len(np.unique(layers_idx)), num_layers)

        super().__init__()

        for ls in layers_slice:
            s, e = ls.split("-")
            for layer_idx in range(int(s), int(e)):
                is_first = layer_idx == min(layers_idx)
                is_last = layer_idx == max(layers_idx)
                module = EMHSResidualLayer(
                    input_nc if is_first else feat_dim,
                    output_nc if is_last else feat_dim,
                    feat_dim,
                    latent_classes,
                    kernel_size,
                )
                self.add_module(str(layer_idx), module)

        print(self)

        def forward(self, x, pos):
            pass
