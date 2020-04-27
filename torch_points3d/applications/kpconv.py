import os
from omegaconf import DictConfig, OmegaConf

from torch_points3d.applications.modelfactory import ModelFactory
from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.modules.KPConv import *
from torch_points3d.core.base_conv.partial_dense import *
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.datasets.multiscale_data import MultiScaleBatch

CUR_FILE = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CONFIG = os.path.join(DIR_PATH, "conf/kpconv")


def KPConv(
    architecture: str = None,
    input_nc: int = None,
    num_layers: int = None,
    in_grid_size: float = 0.02,
    in_feat: int = 64,
    config: DictConfig = None,
    **kwargs
):
    """ Create a KPConv backbone model based on the architecture proposed in
    https://arxiv.org/abs/1904.08889

    Parameters
    ----------
    architecture : str, optional
        Architecture of the model, choose from unet, encoder and decoder
    input_nc : int, optional
        Number of channels for the input
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
        self.resolve_model(model_config)
        modules_lib = sys.modules[__name__]
        return KPConvUnet(model_config, None, None, modules_lib)


class KPConvUnet(UnwrappedUnetBasedModel):
    CONV_TYPE = "partial_dense"

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

    def forward(self, data):
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

        """
        self._set_input(data)
        return super().forward(self.input, precomputed_down=self.pre_computed, precomputed_up=self.upsample)
