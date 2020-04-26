import os
import sys
from omegaconf import DictConfig, OmegaConf

from . import ModelFactory
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
    output_nc: int = None,
    num_layers: int = None,
    config: DictConfig = None,
    multiscale=True,
    **kwargs
):
    factory = PointNet2Factory(
        architecture=architecture,
        num_layers=num_layers,
        input_nc=input_nc,
        output_nc=output_nc,
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

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        Sets:
            self.input:
                x -- Features [B, C, N]
                pos -- Points [B, N, 3]
        """
        assert len(data.pos.shape) == 3
        data = data.to(device)
        if data.x is not None:
            x = data.x.transpose(1, 2).contiguous()
        else:
            x = None
        self.input = Data(x=x, pos=data.pos)
