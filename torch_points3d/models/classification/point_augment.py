import logging
import torch.nn.functional as F
from typing import Any
from torch import nn
from torch_points3d.applications.pointnet2 import PointNet2
from torch_points3d.applications.kpconv import KPConv
from torch_points3d.core.data_transform import AddOnes
from torch_points3d.applications.rsconv import RSConv
from torch_points3d.models.base_model import BaseModel
from torch_points3d.modules.PointAugment import PointAugment
from torch_points3d.utils.model_utils import freeze_params, unfreeze_params

log = logging.getLogger(__name__)


class Classifier(nn.Module):
    def __init__(self, model_opt, input_nc, num_classes):
        super(Classifier, self).__init__()

        self._input_nc = input_nc
        self._num_classes = num_classes
        self._model_opt = model_opt
        self._build()

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def conv_type(self):
        return self._model_opt.conv_type

    def _build(self):
        backbone_builder = globals().copy()[self._model_opt.class_name]
        self._model = backbone_builder(
            architecture="encoder",
            input_nc=self._input_nc,
            in_feat=self._model_opt.in_feat,
            num_layers=self._model_opt.num_layers,
            output_nc=self._num_classes,
            multiscale=True,
        )

    def forward(self, data):
        if self._model_opt.class_name == "KPConv":  # KPConv needs added ones for its x features
            data = AddOnes()(data)
            data.x = (
                torch.cat([data.x, data.ones.to(data.pos.device)], dim=-1)
                if data.x is not None
                else data.ones.to(data.pos.device)
            )
        data_out = self._model(data)
        return F.log_softmax(data_out.x.squeeze(), dim=-1), data_out.x


class PointAugmentedModel(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        BaseModel.__init__(self, option)

        input_nc = dataset.feature_dimension if option.backbone == "kpconv" else 3 + dataset.feature_dimension
        self._model = Classifier(option.backbones[option.backbone], input_nc, dataset.num_classes)
        self._point_augment = PointAugment(option, self._model)
        self.loss_names = ["loss", "loss_pa", "loss_p", "loss_reg", "loss_aug"]

    def set_input(self, data, device):
        self._data = data.to(device)
        self.labels = self._data.y.flatten()

    def forward(self, *args, **kwargs) -> Any:
        self.output = self._point_augment(self._data)

    def backward(self):
        self._point_augment.backward()
        self.extract_current_losses(self._point_augment)
