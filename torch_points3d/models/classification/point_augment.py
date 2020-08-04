import logging
import torch.nn.functional as F
from typing import Any
import torch
from torch_points3d.models.base_model import BaseModel
import torch_points3d.modules.PointAugment as pa_modules
from torch_points3d.utils.model_utils import freeze_params, unfreeze_params

log = logging.getLogger(__name__)

class PointAugmentedModel(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        super(PointAugmentedModel, self).__init__(option)

        # a small λ encourages the augmentor to focus more on the classification with less augmentation on P, and vice versa. 
        # In the paper (all experiments), we set λ = 1 to treat the two terms equally.
        self.aug_weight = option.aug_weight
        # γ is to balance the importance of the terms (we empirically set γ as 10.0)
        self.classifier_weight = option.classifier_weight
        self._num_classes = dataset.num_classes

        # 1 - CREATE BACKBONE MODEL
        input_nc = dataset.feature_dimension
        classifier_option = option.classifier
        classifier_cls = getattr(pa_modules, classifier_option.module_name)
        self.classifier_module = classifier_cls(model_opt=classifier_option, input_nc=input_nc, num_classes=self._num_classes)

        # 2 - CREATE POINTAUGMENT MODEL
        pa_option = option.augmentor
        pa_cls = getattr(pa_modules, pa_option.module_name)
        self.pa_module = pa_cls(config=pa_option, conv_type=option.conv_type)

        self.loss_names = ["loss", "loss_pa", "loss_p", "loss_reg", "loss_aug"]

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        # Forward through backbone model

        self.input = data.to(device)
        self.labels = torch.flatten(data.y).long()

    def forward(self, *args, **kwargs):
        data_augmented = self.pa_module(self.input).contiguous()
        
        self.output, feat = self.classifier_module(self.input)
        
        output_augmented, feat_augmented = self.classifier_module(data_augmented)
        
        self.loss_p = F.nll_loss(self.output, self.labels)
        self.loss_pa = F.nll_loss(output_augmented, self.labels)

        pv = max(1, torch.exp(torch.sum(self.output.exp() * F.one_hot(self.labels, self._num_classes))))
        self.loss_aug = self.loss_pa + self.aug_weight * torch.abs(1 - torch.exp(self.loss_pa - pv * self.loss_p))
        self.loss_reg = F.mse_loss(feat, feat_augmented)
        self.loss = self.loss_pa + self.loss_p + self.classifier_weight * self.loss_reg

    def backward(self):
        # Optimize augmentor using LA = L(P') + λ|1.0 − exp[L(P′) − ρL(P)]|.
        freeze_params(self.classifier_module)
        unfreeze_params(self.pa_module)
        self.loss_aug.backward(retain_graph=True)

        # Optimize classifier using LC = L(P') + L(P) + γ||F_g - F_g'||_2
        freeze_params(self.pa_module)
        unfreeze_params(self.classifier_module)
        self.loss.backward()
