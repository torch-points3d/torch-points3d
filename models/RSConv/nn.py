from typing import Any
import torch
import torch.nn.functional as F

from models.unet_base import UnetBasedModel


class SegmentationModel(UnetBasedModel):
    def __init__(self, option, model_name, num_classes, modules):
        # call the initialization method of UnetBasedModel
        UnetBasedModel.__init__(self, option, model_name, num_classes, modules)

        nn = option.mlp_cls.nn
        self.dropout = option.mlp_cls.get('dropout')
        self.lin1 = torch.nn.Linear(nn[0], nn[1])
        self.lin2 = torch.nn.Linear(nn[2], nn[3])
        self.lin3 = torch.nn.Linear(nn[3], num_classes)

        self.loss_names = ['loss_seg']

    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input = (data.x, data.pos, data.batch)
        self.labels = data.y

    def forward(self) -> Any:
        """Standard forward"""
        x, _, _ = self.model(self.input)
        x = F.relu(self.lin1(x))
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
        self.loss_seg = F.nll_loss(self.output, self.labels)
        self.loss_seg.backward()
