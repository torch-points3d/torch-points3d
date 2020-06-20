import logging
import torch
from typing import Any, NamedTuple

from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.modules.EMHS import initialize_emhs

log = logging.getLogger(__name__)


class EMHSInputs(NamedTuple):
    pos: torch.Tensor
    x: torch.Tensor


class Segmentation_MP(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        """Initialize this model class.
        Parameters:
            opt -- training/test options
        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, option)  # call the initialization method of UnetBasedModel

        initialize_emhs(
            option.layers.generator,
            option.num_layers,
            option.layers.module_names,
            option.layers.layers_slice,
            option.layers.latent_classes,
            option.layers.voxelization,
            option.layers.kernel_size,
            option.layers.feat_dim,
        )

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """

    def forward(self, *args, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_seg.backward()  # calculate gradients of network G w.r.t. loss_G
