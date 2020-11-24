import os
import sys
import queue
from omegaconf import DictConfig, OmegaConf
import logging

from torch_points3d.applications.modelfactory import ModelFactory
from torch_points3d.modules.RSConv import *
from torch_points3d.core.base_conv.dense import DenseFPModule
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.datasets.multiscale_data import MultiScaleBatch
from torch_points3d.core.common_modules.dense_modules import Conv1D
from torch_points3d.core.common_modules.base_modules import Seq
from .utils import extract_output_nc

CUR_FILE = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CONFIG = os.path.join(DIR_PATH, "conf/rsconv")

log = logging.getLogger(__name__)


def RSConv(
    architecture: str = None, input_nc: int = None, num_layers: int = None, config: DictConfig = None, *args, **kwargs
):
    """ Create a RSConv backbone model based on the architecture proposed in
    https://arxiv.org/abs/1904.07601

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
    factory = RSConvFactory(
        architecture=architecture, num_layers=num_layers, input_nc=input_nc, config=config, **kwargs
    )
    return factory.build()


class RSConvFactory(ModelFactory):
    def _build_unet(self):
        if self._config:
            model_config = self._config
        else:
            path_to_model = os.path.join(PATH_TO_CONFIG, "unet_{}.yaml".format(self.num_layers))
            model_config = OmegaConf.load(path_to_model)
        ModelFactory.resolve_model(model_config, self.num_features, self._kwargs)
        modules_lib = sys.modules[__name__]
        return RSConvUnet(model_config, None, None, modules_lib, **self.kwargs)

    def _build_encoder(self):
        if self._config:
            model_config = self._config
        else:
            path_to_model = os.path.join(PATH_TO_CONFIG, "encoder_{}.yaml".format(self.num_layers))
            model_config = OmegaConf.load(path_to_model)
        ModelFactory.resolve_model(model_config, self.num_features, self._kwargs)
        modules_lib = sys.modules[__name__]
        return RSConvEncoder(model_config, None, None, modules_lib, **self.kwargs)


class RSConvBase(UnwrappedUnetBasedModel):
    CONV_TYPE = "dense"

    def __init__(self, model_config, model_type, dataset, modules, *args, **kwargs):
        super(RSConvBase, self).__init__(model_config, model_type, dataset, modules)

        default_output_nc = kwargs.get("default_output_nc", 384)
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
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        Sets:
            self.input:
                x -- Features [B, C, N]
                pos -- Points [B, N, 3]
        """
        assert len(data.pos.shape) == 3
        data = data.to(self.device)
        if data.x is not None:
            data.x = data.x.transpose(1, 2).contiguous()
        else:
            data.x = None
        self.input = data


class RSConvEncoder(RSConvBase):
    def __init__(self, model_config, model_type, dataset, modules, *args, **kwargs):
        try:
            default_output_nc = extract_output_nc(model_config)
        except:
            default_output_nc = -1
            log.warning("Could not resolve number of output channels")
        super().__init__(
            model_config, model_type, dataset, modules, default_output_nc=default_output_nc, *args, **kwargs
        )

    def forward(self, data, *args, **kwargs):
        """ This method does a forward on the Unet

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


class RSConvUnet(RSConvBase):
    def __init__(self, model_config, model_type, dataset, modules, *args, **kwargs):
        try:
            default_output_nc = (
                model_config.innermost[0].nn[-1]
                + model_config.innermost[1].nn[-1]
                + model_config.up_conv.up_conv_nn[-1][-1]
            )
        except:
            default_output_nc = -1
            log.warning("Could not resolve number of output channels")
        super().__init__(
            model_config, model_type, dataset, modules, default_output_nc=default_output_nc, *args, **kwargs
        )

    def forward(self, data, *args, **kwargs):
        """ This method does a forward on the Unet

        Parameters:
        -----------
        data
            A dictionary that contains the data itself and its metadata information. Should contain
                x -- Features [B, N, C]
                pos -- Points [B, N, 3]
        """
        self._set_input(data)
        stack_down = []
        queue_up = queue.Queue()

        data = self.input
        stack_down.append(data)

        for i in range(len(self.down_modules) - 1):
            data = self.down_modules[i](data)
            stack_down.append(data)
        sampling_ids = self._collect_sampling_ids(stack_down)

        data = self.down_modules[-1](data)
        queue_up.put(data)

        assert len(self.inner_modules) == 2, "For this segmentation model, we except 2 distinct inner"
        data_inner = self.inner_modules[0](data)
        data_inner_2 = self.inner_modules[1](stack_down[3])

        for i in range(len(self.up_modules)):
            data = self.up_modules[i]((queue_up.get(), stack_down.pop()))
            queue_up.put(data)

        last_feature = torch.cat(
            [data.x, data_inner.x.repeat(1, 1, data.x.shape[-1]), data_inner_2.x.repeat(1, 1, data.x.shape[-1])], dim=1
        )

        if self.has_mlp_head:
            data.x = self.mlp(last_feature)
        else:
            data.x = last_feature
        for key, value in sampling_ids.items():
            setattr(data, key, value)
        return data
