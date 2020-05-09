import logging
import torch
import torch.nn.functional as F
from typing import Any
from torch import nn
from torch_points3d.applications.pointnet2 import PointNet2
from torch_points3d.core.common_modules.base_modules import Seq
from torch_points3d.core.common_modules.dense_modules import Conv1D
from torch_points3d.models.base_model import BaseModel
from torch_points3d.modules.PointAugment import PointAugment
from torch_points3d.utils.model_utils import freeze_params, unfreeze_params

log = logging.getLogger(__name__)


class ClassificationModel(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        """Initialize this model class.
        Parameters:
            opt -- training/test options
        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, option)

        self._num_classes = dataset.num_classes
        self._model = nn.ModuleDict()
        self._model["backbone"] = PointNet2(
            architecture="encoder",
            input_nc=option.input_nc,
            in_feat=option.in_feat,
            num_layers=option.num_layers,
            output_nc=option.output_nc,
            multiscale=True,
        )
        self._model["classifier"] = Seq()
        self._model["classifier"].append(
            Conv1D(option.output_nc, self._num_classes, activation=None, bias=True, bn=False)
        )
        self._point_augment = PointAugment(dataset.feature_dimension, option.conv_type, option.point_augment)

        self.loss_names = ["loss", "loss_pa", "loss_p", "loss_reg", "loss_aug"]

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self._data = data.to(device)
        self.labels = self._data.y.squeeze()

    def _forward(self, data):
        data_out = self._model["backbone"](data)
        return F.log_softmax(self._model["classifier"](data_out.x).squeeze(), dim=-1), data_out.x

    def forward(self, *args, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        data_augmented = self._point_augment(self._data)

        self.output, feat = self._forward(self._data)
        output_augmented, feat_augmented = self._forward(data_augmented)

        self.loss_p = F.nll_loss(self.output, self.labels)
        self.loss_pa = F.nll_loss(output_augmented, self.labels)

        pv = max(1, torch.exp(torch.sum(self.output.exp() * F.one_hot(self.labels, self._num_classes))))
        self.loss_aug = torch.abs(1 - torch.exp(self.loss_pa - pv * self.loss_p))
        self.loss_reg = F.mse_loss(feat, feat_augmented)
        self.loss = self.loss_pa + self.loss_p + self.loss_reg

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # Optimize augmentor using LA = |1.0 − exp[L(P′) − ρL(P)]|.
        freeze_params(self._model)
        unfreeze_params(self._point_augment)
        self.loss_aug.backward(retain_graph=True)

        # Optimize classifier
        freeze_params(self._point_augment)
        unfreeze_params(self._model)
        self.loss.backward()
