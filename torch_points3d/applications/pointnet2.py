import os
import sys
from omegaconf import DictConfig, OmegaConf
import logging

from torch_points3d.applications.modelfactory import ModelFactory
from torch_points3d.modules.pointnet2 import *
from torch_points3d.core.base_conv.dense import DenseFPModule
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.datasets.multiscale_data import MultiScaleBatch
from torch_points3d.core.common_modules.dense_modules import Conv1D
from torch_points3d.core.common_modules.base_modules import Seq
from .utils import extract_output_nc

CUR_FILE = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CONFIG = os.path.join(DIR_PATH, "conf/pointnet2")

log = logging.getLogger(__name__)


def PointNet2(
    architecture: str = None,
    input_nc: int = None,
    num_layers: int = None,
    config: DictConfig = None,
    multiscale=False,
    *args,
    **kwargs
):
    """ Create a PointNet2 backbone model based on the architecture proposed in
    https://arxiv.org/abs/1706.02413

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
    factory = PointNet2Factory(
        architecture=architecture,
        num_layers=num_layers,
        input_nc=input_nc,
        multiscale=multiscale,
        config=config,
        **kwargs
    )
    return factory.build()


class PointNet2Factory(ModelFactory):
    def _build_unet(self):
        if self._config:
            model_config = self._config
        else:
            path_to_model = os.path.join(
                PATH_TO_CONFIG, "unet_{}_{}.yaml".format(self.num_layers, "ms" if self.kwargs["multiscale"] else "ss")
            )
            model_config = OmegaConf.load(path_to_model)
        ModelFactory.resolve_model(model_config, self.num_features, self._kwargs)
        modules_lib = sys.modules[__name__]
        return PointNet2Unet(model_config, None, None, modules_lib, **self.kwargs)

    def _build_encoder(self):
        if self._config:
            model_config = self._config
        else:
            path_to_model = os.path.join(
                PATH_TO_CONFIG,
                "encoder_{}_{}.yaml".format(self.num_layers, "ms" if self.kwargs["multiscale"] else "ss"),
            )
            model_config = OmegaConf.load(path_to_model)
        ModelFactory.resolve_model(model_config, self.num_features, self._kwargs)
        modules_lib = sys.modules[__name__]
        return PointNet2Encoder(model_config, None, None, modules_lib, **self.kwargs)


class BasePointnet2(UnwrappedUnetBasedModel):

    CONV_TYPE = "dense"

    def __init__(self, model_config, model_type, dataset, modules, *args, **kwargs):
        super(BasePointnet2, self).__init__(model_config, model_type, dataset, modules)

        try:
            default_output_nc = extract_output_nc(model_config)
        except:
            default_output_nc = -1
            log.warning("Could not resolve number of output channels")

        self._has_mlp_head = False
        self._output_nc = default_output_nc
        if "output_nc" in kwargs:
            self._has_mlp_head = True
            self._output_nc = kwargs["output_nc"]
            self.mlp = Seq()
            self.mlp.append(Conv1D(default_output_nc, self._output_nc, bn=True, bias=False))

    @property
    def has_mlp_head(self):
        return self._has_mlp_head

    @property
    def output_nc(self):
        return self._output_nc

    def _set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """
        assert len(data.pos.shape) == 3
        data = data.to(self.device)
        if data.x is not None:
            data.x = data.x.transpose(1, 2).contiguous()
        else:
            data.x = None
        self.input = data


class PointNet2Encoder(BasePointnet2):
    def forward(self, data, *args, **kwargs):
        """
        Parameters:
        -----------
        data
            A dictionary that contains the data itself and its metadata information. Should contain
                x -- Features [B, N, C]
                pos -- Points [B, N, 3]
        """
        self._set_input(data)
        data = self.input
        stack_down = [data]
        for i in range(len(self.down_modules) - 1):
            data = self.down_modules[i](data)
            stack_down.append(data)
        data = self.down_modules[-1](data)

        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append(data)
            data = self.inner_modules[0](data)

        if self.has_mlp_head:
            data.x = self.mlp(data.x)
        return data


class PointNet2Unet(BasePointnet2):
    def forward(self, data, *args, **kwargs):
        """ This method does a forward on the Unet assuming symmetrical skip connections
        Input --- D1 -- D2 -- I -- U1 -- U2 -- U3 -- output
           |       |      |________|      |    |
           |       |______________________|    |
           |___________________________________|

        Parameters:
        -----------
        data
            A dictionary that contains the data itself and its metadata information. Should contain
                x -- Features [B, N, C]
                pos -- Points [B, N, 3]
        """
        self._set_input(data)
        data = self.input
        stack_down = [data]
        for i in range(len(self.down_modules) - 1):
            data = self.down_modules[i](data)
            stack_down.append(data)
        data = self.down_modules[-1](data)

        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append(data)
            data = self.inner_modules[0](data)

        sampling_ids = self._collect_sampling_ids(stack_down)

        for i in range(len(self.up_modules)):
            data = self.up_modules[i]((data, stack_down.pop()))

        for key, value in sampling_ids.items():
            setattr(data, key, value)

        if self.has_mlp_head:
            data.x = self.mlp(data.x)

        return data
