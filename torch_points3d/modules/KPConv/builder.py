import os
import logging
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch_points3d.utils.model_building_utils.model_definition_resolver import _resolve

from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.modules.KPConv import *
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.applications import ModelFactory

current_folder = os.path.dirname(os.path.realpath(__file__))

log = logging.getLogger(__name__)


def build_unet(model_factory: ModelFactory):

    num_layers = model_factory.num_layers
    num_features = model_factory.num_features
    kwargs = model_factory.kwargs

    path_to_model = os.path.join(current_folder, "api/unet_{}.yaml".format(num_layers))
    model_config = OmegaConf.load(path_to_model)
    model_factory.resolve_model(model_config, num_features, kwargs)

    UnwrappedUnetBasedModel.__init__(
        model_factory, model_config, model_config.conv_type, None, model_factory.modules_lib
    )
