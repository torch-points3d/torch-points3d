import logging
import torch.nn.functional as F
from typing import Any
from torch import nn
from torch_points3d.applications.pointnet2 import PointNet2
from torch_points3d.models.base_model import BaseModel
from torch_points3d.modules.PointAugment import PointAugment
from torch_points3d.utils.model_utils import freeze_params, unfreeze_params

log = logging.getLogger(__name__)


class Classifier(nn.Module):
    def __init__(self, option, num_classes):
        super(Classifier, self).__init__()
        self._model = PointNet2(
            architecture="encoder",
            input_nc=option.input_nc,
            in_feat=option.in_feat,
            num_layers=option.num_layers,
            output_nc=num_classes,
            multiscale=True,
        )

    def forward(self, data):
        data_out = self._model(data)
        return F.log_softmax(data_out.x.squeeze(), dim=-1), data_out.x


class PointAugmentedPointnet2(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        BaseModel.__init__(self, option)
        self._model = Classifier(option, dataset.num_classes)
        self._point_augment = PointAugment(
            self._model, dataset.feature_dimension, dataset.num_classes, option.conv_type, option.point_augment
        )
        self.loss_names = ["loss", "loss_pa", "loss_p", "loss_reg", "loss_aug"]

    def set_input(self, data, device):
        self._data = data.to(device)
        self.labels = self._data.y.flatten()

    def forward(self, *args, **kwargs) -> Any:
        self.output = self._point_augment(self._data)

    def backward(self):
        self._point_augment.backward()
        self.extract_current_losses(self._point_augment)
