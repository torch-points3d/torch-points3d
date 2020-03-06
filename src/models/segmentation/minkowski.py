import etw_pytorch_utils as pt_utils
import logging

from src.modules.MinkowskiEngine import *
from src.models.base_architectures import UnwrappedUnetBasedModel

log = logging.getLogger(__name__)


class Minkowski_Model(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnetBasedModel
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)

    def set_input(self, data, device):
        coords = torch.cat([data.indices, data.batch.unsqueeze(-1)], -1).int()
        self.input = ME.SparseTensor(data.x, coords=coords).to(device)
        self.labels = data.y

    def forward(self):

        stack_down = []

        x = self.input
        for i in range(len(self.down_modules) - 1):
            print(x.shape)
            x = self.down_modules[i](x)
            stack_down.append(x)

        x = self.down_modules[-1](x)

        for i in range(len(self.up_modules)):
            x = self.up_modules[i](x, stack_down.pop())

        import pdb

        pdb.set_trace()

    def backward(self):
        pass
