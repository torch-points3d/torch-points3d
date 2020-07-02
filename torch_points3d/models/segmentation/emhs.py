import logging
import torch.nn.functional as F
import torch

from torch_points3d.modules.MinkowskiEngine import *
from torch_points3d.models.base_architectures import UnwrappedUnetBasedModel
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL


log = logging.getLogger(__name__)


class EMHS_Model(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        super(EMHS_Model, self).__init__(option)
        self.model = initialize_minkowski_emls(
            model_name=option.model_name,
            input_nc=dataset.feature_dimension,
            dim_feat=option.dim_feat,
            output_nc=dataset.num_classes,
            num_layer=option.num_layer,
            kernel_size=option.kernel_size,
            stride=option.stride,
            dilation=option.dilation,
            D=option.D,
        )
        self.loss_names = ["loss_seg"]

    def set_input(self, data, device):

        self.batch_idx = data.batch.squeeze()
        coords = torch.cat([data.batch.unsqueeze(-1).int(), data.coords.int()], -1)
        self.input = ME.SparseTensor(data.x, coords=coords).to(device)
        self.labels = data.y.to(device)

    def forward(self, *args, **kwargs):
        self.output = F.log_softmax(self.model(self.input).feats, dim=-1)
        self.loss_seg = F.nll_loss(self.output, self.labels, ignore_index=IGNORE_LABEL)

    def backward(self):
        self.loss_seg.backward()
