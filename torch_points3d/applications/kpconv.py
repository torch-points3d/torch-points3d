import os
from omegaconf import DictConfig, OmegaConf
import logging

from torch_points3d.applications.modelfactory import ModelFactory
from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.modules.KPConv import *
from torch_points3d.core.base_conv.partial_dense import *
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.datasets.multiscale_data import MultiScaleBatch
from torch_points3d.core.common_modules.base_modules import MLP
from .utils import extract_output_nc


CUR_FILE = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CONFIG = os.path.join(DIR_PATH, "conf/kpconv")

log = logging.getLogger(__name__)


def KPConv(
    architecture: str = None, input_nc: int = None, num_layers: int = None, config: DictConfig = None, *args, **kwargs
):
    """ Create a KPConv backbone model based on the architecture proposed in
    https://arxiv.org/abs/1904.08889

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
    in_grid_size : float, optional
        Size of the grid at the entry of the network. It is divided by two at each layer
    in_feat : int, optional
        Number of channels after the first convolution. Doubles at each layer
    config : DictConfig, optional
        Custom config, overrides the num_layers and architecture parameters
    """
    factory = KPConvFactory(
        architecture=architecture, num_layers=num_layers, input_nc=input_nc, config=config, **kwargs
    )
    return factory.build()


class KPConvFactory(ModelFactory):
    def _build_unet(self):
        if self._config:
            model_config = self._config
        else:
            path_to_model = os.path.join(PATH_TO_CONFIG, "unet_{}.yaml".format(self.num_layers))
            model_config = OmegaConf.load(path_to_model)
        ModelFactory.resolve_model(model_config, self.num_features, self._kwargs)
        modules_lib = sys.modules[__name__]
        return KPConvUnet(model_config, None, None, modules_lib, **self.kwargs)

    def _build_encoder(self):
        if self._config:
            model_config = self._config
        else:
            path_to_model = os.path.join(PATH_TO_CONFIG, "encoder_{}.yaml".format(self.num_layers))
            model_config = OmegaConf.load(path_to_model)
        ModelFactory.resolve_model(model_config, self.num_features, self._kwargs)
        modules_lib = sys.modules[__name__]
        return KPConvEncoder(model_config, None, None, modules_lib, **self.kwargs)


class BaseKPConv(UnwrappedUnetBasedModel):
    CONV_TYPE = "partial_dense"

    def __init__(self, model_config, model_type, dataset, modules, *args, **kwargs):
        super(BaseKPConv, self).__init__(model_config, model_type, dataset, modules)
        try:
            default_output_nc = extract_output_nc(model_config)
        except:
            default_output_nc = -1
            log.warning("Could not resolve number of output channels")

        self._output_nc = default_output_nc
        self._has_mlp_head = False
        if "output_nc" in kwargs:
            self._has_mlp_head = True
            self._output_nc = kwargs["output_nc"]
            self.mlp = MLP([default_output_nc, self.output_nc], activation=torch.nn.LeakyReLU(0.2), bias=False)

    @property
    def has_mlp_head(self):
        return self._has_mlp_head

    @property
    def output_nc(self):
        return self._output_nc

    def _set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters
        -----------
        data:
            a dictionary that contains the data itself and its metadata information.
        """
        data = data.to(self.device)
        if isinstance(data, MultiScaleBatch):
            self.pre_computed = data.multiscale
            self.upsample = data.upsample
            del data.upsample
            del data.multiscale
        else:
            self.upsample = None
            self.pre_computed = None

        self.input = data


class KPConvEncoder(BaseKPConv):
    def forward(self, data, *args, **kwargs):
        """
        Parameters
        -----------
        data:
            A dictionary that contains the data itself and its metadata information. Should contain
            - pos [N, 3]
            - x [N, C]
            - multiscale (optional) precomputed data for the down convolutions
            - upsample (optional) precomputed data for the up convolutions

        Returns
        --------
        data:
            - pos [1, 3] - Dummy pos
            - x [1, output_nc]
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


class KPConvUnet(BaseKPConv):
    def forward(self, data, *args, **kwargs):
        """Run forward pass.
        Input --- D1 -- D2 -- D3 -- U1 -- U2 -- output
                   |      |_________|     |
                   |______________________|

        Parameters
        -----------
        data:
            A dictionary that contains the data itself and its metadata information. Should contain
            - pos [N, 3]
            - x [N, C]
            - multiscale (optional) precomputed data for the down convolutions
            - upsample (optional) precomputed data for the up convolutions

        Returns
        --------
        data:
            - pos [N, 3]
            - x [N, output_nc]
        """
        self._set_input(data)
        data = super().forward(self.input, precomputed_down=self.pre_computed, precomputed_up=self.upsample)
        if self.has_mlp_head:
            data.x = self.mlp(data.x)
        return data
