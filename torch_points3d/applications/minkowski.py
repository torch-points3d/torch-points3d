import os
from omegaconf import DictConfig, OmegaConf
import logging
import torch
from torch_geometric.data import Batch

from torch_points3d.applications.modelfactory import ModelFactory
from torch_points3d.modules.MinkowskiEngine import *
from torch_points3d.core.base_conv.message_passing import *
from torch_points3d.core.base_conv.partial_dense import *
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.core.common_modules.base_modules import MLP

from .utils import extract_output_nc


CUR_FILE = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CONFIG = os.path.join(DIR_PATH, "conf/minkowski")

log = logging.getLogger(__name__)


def Minkowski(
    architecture: str = None, input_nc: int = None, num_layers: int = None, config: DictConfig = None, *args, **kwargs
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
        architecture=architecture, num_layers=num_layers, input_nc=input_nc, config=config, **kwargs
    )
    return factory.build()


class MinkowskiFactory(ModelFactory):
    def _build_unet(self):
        if self._config:
            model_config = self._config
        else:
            path_to_model = os.path.join(PATH_TO_CONFIG, "unet_{}.yaml".format(self.num_layers))
            model_config = OmegaConf.load(path_to_model)
        ModelFactory.resolve_model(model_config, self.num_features, self._kwargs)
        modules_lib = sys.modules[__name__]
        return MinkowskiUnet(model_config, None, None, modules_lib, **self.kwargs)

    def _build_encoder(self):
        if self._config:
            model_config = self._config
        else:
            path_to_model = os.path.join(PATH_TO_CONFIG, "encoder_{}.yaml".format(self.num_layers),)
            model_config = OmegaConf.load(path_to_model)
        ModelFactory.resolve_model(model_config, self.num_features, self._kwargs)
        modules_lib = sys.modules[__name__]
        return MinkowskiEncoder(model_config, None, None, modules_lib, **self.kwargs)


class BaseMinkowski(UnwrappedUnetBasedModel):
    CONV_TYPE = "sparse"

    def __init__(self, model_config, model_type, dataset, modules, *args, **kwargs):
        super(BaseMinkowski, self).__init__(model_config, model_type, dataset, modules)
        self.weight_initialization()
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

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters
        -----------
        data:
            a dictionary that contains the data itself and its metadata information.
        """
        coords = torch.cat([data.batch.unsqueeze(-1).int(), data.coords.int()], -1)
        self.input = ME.SparseTensor(data.x, coords=coords).to(self.device)
        if data.pos is not None:
            self.xyz = data.pos.to(self.device)
        else:
            self.xyz = data.coords.to(self.device)


class MinkowskiEncoder(BaseMinkowski):
    def forward(self, data, *args, **kwargs):
        """
        Parameters:
        -----------
        data
            A SparseTensor that contains the data itself and its metadata information. Should contain
                F -- Features [N, C]
                coords -- Coords [N, 4]

        Returns
        --------
        data:
            - x [1, output_nc]

        """
        self._set_input(data)
        data = self.input
        for i in range(len(self.down_modules)):
            data = self.down_modules[i](data)

        out = Batch(x=data.F, batch=data.C[:, 0].long().to(data.F.device))
        if not isinstance(self.inner_modules[0], Identity):
            out = self.inner_modules[0](out)

        if self.has_mlp_head:
            out.x = self.mlp(out.x)
        return out


class MinkowskiUnet(BaseMinkowski):
    def forward(self, data, *args, **kwargs):
        """Run forward pass.
        Input --- D1 -- D2 -- D3 -- U1 -- U2 -- output
                   |      |_________|     |
                   |______________________|

        Parameters
        -----------
        data
            A SparseTensor that contains the data itself and its metadata information. Should contain
                F -- Features [N, C]
                coords -- Coords [N, 4]

        Returns
        --------
        data:
            - pos [N, 3] (coords or real pos if xyz is in data)
            - x [N, output_nc]
            - batch [N]
        """
        self._set_input(data)
        data = self.input
        stack_down = []
        for i in range(len(self.down_modules) - 1):
            data = self.down_modules[i](data)
            stack_down.append(data)

        data = self.down_modules[-1](data)
        stack_down.append(None)
        # TODO : Manage the inner module
        for i in range(len(self.up_modules)):
            data = self.up_modules[i](data, stack_down.pop())

        out = Batch(x=data.F, pos=self.xyz, batch=data.C[:, 0])
        if self.has_mlp_head:
            out.x = self.mlp(out.x)
        return out
