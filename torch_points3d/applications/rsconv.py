import os
import sys
import queue
from omegaconf import DictConfig, OmegaConf

from torch_points3d.applications.modelfactory import ModelFactory
from torch_points3d.modules.RSConv import *
from torch_points3d.core.base_conv.dense import DenseFPModule
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.datasets.multiscale_data import MultiScaleBatch

CUR_FILE = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CONFIG = os.path.join(DIR_PATH, "conf/rsconv")


def RSConv(architecture: str = None, input_nc: int = None, num_layers: int = None, config: DictConfig = None, **kwargs):
    """ Create a RSConv backbone model based on the architecture proposed in
    https://arxiv.org/abs/1904.07601

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
        self.resolve_model(model_config)
        modules_lib = sys.modules[__name__]
        return RSConvUnet(model_config, None, None, modules_lib)


class RSConvUnet(UnwrappedUnetBasedModel):
    CONV_TYPE = "dense"

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
            x = data.x.transpose(1, 2).contiguous()
        else:
            x = None
        self.input = Data(x=x, pos=data.pos)

    def forward(self, data):
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

        data = self.down_modules[-1](data)
        queue_up.put(data)

        assert len(self.inner_modules) == 2, "For this segmentation model, we except 2 distinct inner"
        data_inner = self.inner_modules[0](data)
        data_inner_2 = self.inner_modules[1](stack_down[3])

        for i in range(len(self.up_modules) - 1):
            data = self.up_modules[i]((queue_up.get(), stack_down.pop()))
            queue_up.put(data)

        last_feature = torch.cat(
            [data.x, data_inner.x.repeat(1, 1, data.x.shape[-1]), data_inner_2.x.repeat(1, 1, data.x.shape[-1])], dim=1
        )
        data.x = last_feature
        return data
