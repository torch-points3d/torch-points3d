import os
import sys
from omegaconf import DictConfig, OmegaConf
import logging
import torch
from torch_geometric.data import Batch

from torch_points3d.applications.modelfactory import ModelFactory
import torch_points3d.modules.SparseConv3d as sp3d
from torch_points3d.core.base_conv.message_passing import *
from torch_points3d.modules.SparseConv3d.modules import *
from torch_points3d.core.base_conv.partial_dense import *
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.core.common_modules.base_modules import MLP

from .utils import extract_output_nc


CUR_FILE = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CONFIG = os.path.join(DIR_PATH, "conf/sparseconv3d")

log = logging.getLogger(__name__)


def SparseConv3d(
    architecture: str = None,
    input_nc: int = None,
    num_layers: int = None,
    config: DictConfig = None,
    backend: str = "minkowski",
    *args,
    **kwargs
):
    """Create a Sparse Conv backbone model based on architecture proposed in
     https://arxiv.org/abs/1904.08755

     Two backends are available at the moment:
         - https://github.com/mit-han-lab/torchsparse
         - https://github.com/NVIDIA/MinkowskiEngine

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
     block:
         Type of resnet block, ResBlock by default but can be any of the blocks in modules/SparseConv3d/modules.py
     backend:
         torchsparse or minkowski
    """
    if "SPARSE_BACKEND" in os.environ and sp3d.nn.backend_valid(os.environ["SPARSE_BACKEND"]):
        sp3d.nn.set_backend(os.environ["SPARSE_BACKEND"])
    else:
        sp3d.nn.set_backend(backend)
    
    factory = SparseConv3dFactory(
        architecture=architecture, num_layers=num_layers, input_nc=input_nc, config=config, **kwargs
    )
    return factory.build()


class SparseConv3dFactory(ModelFactory):
    def _build_unet(self):
        if self._config:
            model_config = self._config
        else:
            path_to_model = os.path.join(PATH_TO_CONFIG, "unet_{}.yaml".format(self.num_layers))
            model_config = OmegaConf.load(path_to_model)
        ModelFactory.resolve_model(model_config, self.num_features, self._kwargs)
        modules_lib = sys.modules[__name__]
        return SparseConv3dUnet(model_config, None, None, modules_lib, **self.kwargs)

    def _build_encoder(self):
        if self._config:
            model_config = self._config
        else:
            path_to_model = os.path.join(
                PATH_TO_CONFIG,
                "encoder_{}.yaml".format(self.num_layers),
            )
            model_config = OmegaConf.load(path_to_model)
        ModelFactory.resolve_model(model_config, self.num_features, self._kwargs)
        modules_lib = sys.modules[__name__]
        return SparseConv3dEncoder(model_config, None, None, modules_lib, **self.kwargs)


class BaseSparseConv3d(UnwrappedUnetBasedModel):
    CONV_TYPE = "sparse"

    def __init__(self, model_config, model_type, dataset, modules, *args, **kwargs):
        super().__init__(model_config, model_type, dataset, modules)
        self.weight_initialization()
        default_output_nc = kwargs.get("default_output_nc", None)
        if not default_output_nc:
            default_output_nc = extract_output_nc(model_config)

        self._output_nc = default_output_nc
        self._has_mlp_head = False
        if "output_nc" in kwargs:
            self._has_mlp_head = True
            self._output_nc = kwargs["output_nc"]
            self.mlp = MLP([default_output_nc, self.output_nc], activation=torch.nn.ReLU(), bias=False)

    @property
    def has_mlp_head(self):
        return self._has_mlp_head

    @property
    def output_nc(self):
        return self._output_nc

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, sp3d.nn.Conv3d) or isinstance(m, sp3d.nn.Conv3dTranspose):
                torch.nn.init.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, sp3d.nn.BatchNorm):
                torch.nn.init.constant_(m.bn.weight, 1)
                torch.nn.init.constant_(m.bn.bias, 0)

    def _set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters
        -----------
        data:
            a dictionary that contains the data itself and its metadata information.
        """
        self.input = sp3d.nn.SparseTensor(data.x, data.coords, data.batch, self.device)
        if data.pos is not None:
            self.xyz = data.pos
        else:
            self.xyz = data.coords

class SparseConv3dEncoder(BaseSparseConv3d):
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


class SparseConv3dUnet(BaseSparseConv3d):
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

        out = Batch(x=data.F, pos=self.xyz).to(self.device)
        if self.has_mlp_head:
            out.x = self.mlp(out.x)
        return out
