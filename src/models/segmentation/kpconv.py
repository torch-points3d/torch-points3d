from .base import Segmentation_MP
from src.modules.KPConv import *
from src.models.base_model import BaseModel


class KPConvSeg(Segmentation_MP):
    """ Basic implementation of KPConv"""

    def set_input(self, data):
        self.input = data
        self.batch_idx = data.batch
        self.labels = data.y


class OfficialKPConv(BaseModel):
    def __init__(self, cfg):
        pass

    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input = data
        self.labels = data.y

    def forward(self) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        data = self.model(self.input)
        x = F.relu(self.lin1(data.x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        self.output = F.log_softmax(x, dim=-1)
        return self.output

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results

        self.loss_seg = F.nll_loss(self.output, self.labels) + self.get_internal_loss()

        if torch.isnan(self.loss_seg):
            import pdb

            pdb.set_trace()
        self.loss_seg.backward()  # calculate gradients of network G w.r.t. loss_G
