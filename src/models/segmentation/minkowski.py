import etw_pytorch_utils as pt_utils
import logging

from src.modules.MinkowskiEngine import *
from src.models.base_architectures import UnwrappedUnetBasedModel

log = logging.getLogger(__name__)


class Minkowski_Model(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnetBasedModel
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)

    def set_input(self, data):

        self.input = ME.SparseTensor(data.x, coords=data.indices).to(data.x.device)
        self.labels = data.y

    def forward(self):

        stack_down = []

        data = self.input
        for i in range(len(self.down_modules) - 1):
            data = self.down_modules[i](data, pre_computed=self.pre_computed)
            stack_down.append(data)

        data = self.down_modules[-1](data, pre_computed=self.pre_computed)
        innermost = False

        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append(data)
            data = self.inner_modules[0](data)
            innermost = True

        for i in range(len(self.up_modules)):
            if i == 0 and innermost:
                data = self.up_modules[i]((data, stack_down.pop()))
            else:
                data = self.up_modules[i]((data, stack_down.pop()), precomputed_up=self.upsample)

        import pdb

        pdb.set_trace()

    def backward(self):
        pass
