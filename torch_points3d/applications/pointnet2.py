import os
import sys
from omegaconf import DictConfig, OmegaConf

from torch_points3d.applications.modelfactory import ModelFactory
from torch_points3d.modules.pointnet2 import *
from torch_points3d.core.base_conv.dense import DenseFPModule
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.datasets.multiscale_data import MultiScaleBatch

CUR_FILE = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CONFIG = os.path.join(DIR_PATH, "conf/pointnet2")


def PointNet2(
    architecture: str = None,
    input_nc: int = None,
    num_layers: int = None,
    config: DictConfig = None,
    multiscale=True,
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
        self.resolve_model(model_config)
        modules_lib = sys.modules[__name__]
        return PointNet2Unet(model_config, None, None, modules_lib)


class PointNet2Unet(UnwrappedUnetBasedModel):
    CONV_TYPE = "dense"

    def _set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """
        assert len(data.pos.shape) == 3
        data = data.to(self.device)
        if data.x is not None:
            x = data.x.transpose(1, 2).contiguous()
        else:
            x = None
        self.input = Data(x=x, pos=data.pos)

    def forward(self, data):
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

        for i in range(len(self.up_modules)):
            data = self.up_modules[i]((data, stack_down.pop()))
        return data
