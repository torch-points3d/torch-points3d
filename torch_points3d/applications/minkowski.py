import os
from omegaconf import DictConfig, OmegaConf
import logging

from torch_points3d.applications.modelfactory import ModelFactory
from torch_points3d.modules.MinkowskiEngine import *
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel

from .utils import extract_output_nc


CUR_FILE = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CONFIG = os.path.join(DIR_PATH, "conf/minkowski")

log = logging.getLogger(__name__)


def Minkowski(
    architecture: str = None,
    input_nc: int = None,
    num_layers: int = None,
    config: DictConfig = None,
    multiscale=True,
    *args,
    **kwargs
):
    """ Create a Minkowski backbone model based on architecture proposed in
    https://arxiv.org/abs/1904.08755

    Parameters
    ----------
    architecture : str, optional
        Architecture of the model, choose from unet, encoder and decoder
    input_nc : int, optional
        Number of channels for the input
   output_nc : int, optional
        If specified, then we add a fully connected head at the end of the network to provide the requested dimension
    num_layers : int, optional
        Depth of the network
    config : DictConfig, optional
        Custom config, overrides the num_layers and architecture parameters
    """

    factory = MinkowskiFactory(
        architecture=architecture,
        num_layers=num_layers,
        input_nc=input_nc,
        multiscale=multiscale,
        config=config,
        **kwargs
    )
    return factory.build()


class MinkowskiFactory(ModelFactory):
    def _build_unet(self):
        if self._config:
            model_config = self._config
        else:
            path_to_model = os.path.join(
                PATH_TO_CONFIG, "unet_{}_{}.yaml".format(self.num_layers, "ms" if self.kwargs["multiscale"] else "ss")
            )
            model_config = OmegaConf.load(path_to_model)
        self.resolve_model(model_config)
        modules_lib = sys.modules[__name__]
        return MinkowskiUnet(model_config, None, None, modules_lib, **self.kwargs)

    def _build_encoder(self):
        if self._config:
            model_config = self._config
        else:
            path_to_model = os.path.join(
                PATH_TO_CONFIG,
                "encoder_{}_{}.yaml".format(self.num_layers, "ms" if self.kwargs["multiscale"] else "ss"),
            )
            model_config = OmegaConf.load(path_to_model)
        self.resolve_model(model_config)
        modules_lib = sys.modules[__name__]
        return MinkowskiEncoder(model_config, None, None, modules_lib, **self.kwargs)


class BaseMinkowski(UnwrappedUnetBasedModel):
    CONV_TYPE = "sparse"


class MinkowskiUnet(BaseMinkowski):
    pass


class MinkowskiEncoder(BaseMinkowski):
    pass
