import logging
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.modules.KPConv import *
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.api import ModelFactory

log = logging.getLogger(__name__)


def build_unet(model_factory: ModelFactory):
    print(model_factory)
