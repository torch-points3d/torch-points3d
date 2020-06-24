import logging
from typing import Any, NamedTuple
import torch
import torch.nn.functional as F
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.modules.EMHS import initialize_emhs
from torch_points3d.core.data_transform import GridSampling3DIdx

log = logging.getLogger(__name__)


class EMHSInputs(NamedTuple):
    pos: torch.Tensor
    x: torch.Tensor


class EMHSPaper(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        """Initialize this model class.
        Parameters:
            opt -- training/test options
        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, option)  # call the initialization method of UnetBasedModel

        self.model = initialize_emhs(
            option.layers.model_name,
            dataset.feature_dimension,
            dataset.num_classes,
            option.num_layers,
            option.layers.num_elm,
            option.layers.use_attention,
            option.layers.layers_slice,
            option.layers.latent_classes,
            option.layers.voxelization,
            option.layers.kernel_size,
            option.layers.feat_dim,
            option.layers.attention_type,
        )

        self._gr = GridSampling3DIdx(option.layers.voxelization.to_container())

        self.loss_names = ["loss_seg"]

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input = self._gr(data.to(device))
        self.labels = data.y.to(device)

    def forward(self, *args, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.output = F.log_softmax(
            self.model.forward(
                self.input.x, self.input.consecutive_cluster, self.input.cluster_non_consecutive, batch=self.input.batch
            ),
            dim=-1,
        )

        self.loss_seg = F.nll_loss(self.output, self.labels, ignore_index=IGNORE_LABEL)

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_seg.backward()  # calculate gradients of network G w.r.t. loss_G
