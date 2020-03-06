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
        pass

    def forward(self):
        pass

    def backward(self):
        pass
