import torch

from torch_points3d.modules.MinkowskiEngine import *
from torch_points3d.models.base_architectures import UnwrappedUnetBasedModel
from torch_points3d.models.registration.minkowski import BaseMinkowski
from torch_points3d.core.common_modules import FastBatchNorm1d, Seq


class EMHS_Model(BaseMinkowski):
    def __init__(self, option, model_type, dataset, modules):
        BaseMinkowski.__init__(self, option, model_type, dataset, modules)

        self.model = initialize_minkowski_emls(
            dataset.feature_dimension,
            dim_feat=option.dim_feat,
            output_nc=option.output_nc,
            num_layer=option.num_layer,
            kernel_size=option.kernel_size,
            stride=option.stride,
            dilation=option.dilation,
            D=option.D,
        )

    def apply_nn(self, input):
        output = self.model(input).F
        output = self.FC_layer(output)
        if self.normalize_feature:
            return output / (torch.norm(output, p=2, dim=1, keepdim=True) + 1e-3)
        else:
            return output
