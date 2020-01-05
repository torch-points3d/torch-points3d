from typing import Any
import torch
torch.backends.cudnn.enabled = False
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import knn_interpolate
from torch_geometric.nn import radius, global_max_pool

from .modules import SADenseModule
from models.unet_base import UnetBasedModel


class SegmentationModel(UnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnetBasedModel
        UnetBasedModel.__init__(self, option, model_type, dataset, modules)

        nn = option.mlp_cls.nn
        self.dropout = option.mlp_cls.get('dropout')
        self.lin1 = torch.nn.Linear(nn[0], nn[1])
        self.lin2 = torch.nn.Linear(nn[1], nn[2])
        self.lin3 = torch.nn.Linear(nn[2], dataset.num_classes)

        self.loss_names = ['loss_seg']

    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
            Dimensions: [B, N, ...] 
        """
        self.input = data
        self.labels = torch.flatten(data.y)

    def forward(self) -> Any:
        """Standard forward"""
        data = self.model(self.input)
        x = data.x.squeeze(-1)
        x = x.view((-1, x.shape[1]))
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        self.output = x
        return self.output

    def backward(self, debug=False):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results

        if debug:
            print(self.output, torch.isnan(self.output).any(), torch.unique(self.labels))
            print(self.output.shape, self.labels.shape)

            try:
                self.loss_seg = F.cross_entropy(self.output, self.labels.long())
                if torch.isnan(self.loss_seg):
                    import pdb
                    pdb.set_trace()
                self.loss_seg.backward()
            except:
                import pdb
                pdb.set_trace()
            grad_ = self.model.down._local_nn[0].conv.weight.grad
            print(torch.sum(grad_) != 0)
        else:
            self.loss_seg = F.cross_entropy(self.output, self.labels.long())
            self.loss_seg.backward()
