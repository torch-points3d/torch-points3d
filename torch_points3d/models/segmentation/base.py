import logging
import torch
import torch.nn.functional as F
from typing import Any

from torch_points3d.models.base_architectures import UnetBasedModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL

log = logging.getLogger(__name__)


class Segmentation_MP(UnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        """Initialize this model class.
        Parameters:
            opt -- training/test options
        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        UnetBasedModel.__init__(
            self, option, model_type, dataset, modules
        )  # call the initialization method of UnetBasedModel

        self._weight_classes = dataset.weight_classes

        nn = option.mlp_cls.nn
        self.dropout = option.mlp_cls.get("dropout")
        self.lin1 = torch.nn.Linear(nn[0], nn[1])
        self.lin2 = torch.nn.Linear(nn[2], nn[3])
        self.lin3 = torch.nn.Linear(nn[4], dataset.num_classes)

        self.loss_names = ["loss_seg"]

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        data = data.to(device)
        self.input = data
        self.labels = data.y
        self.batch_idx = data.batch

    def forward(self, *args, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        data = self.model(self.input)
        x = F.relu(self.lin1(data.x))
        x = F.dropout(x, p=self.dropout, training=bool(self.training))
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=bool(self.training))
        x = self.lin3(x)
        self.output = F.log_softmax(x, dim=-1)

        if self.labels is not None:
            self.loss_seg = F.nll_loss(self.output, self.labels, ignore_index=IGNORE_LABEL) + self.get_internal_loss()

        return self.output

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_seg.backward()  # calculate gradients of network G w.r.t. loss_G
