import os
from omegaconf import DictConfig, OmegaConf

from . import ModelFactory

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CONFIG = os.path.join(DIR_PATH, "conf/kpconv")


def KPConv(
    architecture: str = None,
    input_nc: int = None,
    output_nc: int = None,
    num_layers: int = None,
    config: DictConfig = None,
    **kwargs
):
    factory = KPConvFactory(
        architecture=architecture,
        num_layers=num_layers,
        input_nc=input_nc,
        output_nc=output_nc,
        config=config,
        **kwargs
    )
    return factory.build()


class KPConvFactory(ModelFactory):
    def _build_unet(self):
        path_to_model = os.path.join(PATH_TO_CONFIG, "unet_{}.yaml".format(self.num_layers))
        model_config = OmegaConf.load(path_to_model)
        self.resolve_model(model_config)
        return KPConvUnet(model_config, model_config.conv_type, None, self.modules_lib)


class KPConvUnet(UnwrappedUnetBasedModel):
    CONV_TYPE = "partial_dense"

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters
        -----------
        data:
            a dictionary that contains the data itself and its metadata information.
        device
            Device on which to run the code. cpu or cuda
        """
        data = data.to(device)

        if isinstance(data, MultiScaleBatch):
            self.pre_computed = data.multiscale
            self.upsample = data.upsample
            del data.upsample
            del data.multiscale
        else:
            self.upsample = None
            self.pre_computed = None

        self.input = data
        self.labels = data.y
        self.batch_idx = data.batch

    def forward(self):
        """Run forward pass."""

        stack_down = []

        data = self.input
        for i in range(len(self.down_modules) - 1):
            data = self.down_modules[i](data, pre_computed=self.pre_computed)
            stack_down.append(data)

        data = self.down_modules[-1](data, pre_computed=self.pre_computed)
        innermost = False

        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append(data)
            data = self.inner_modules[0](data)
            innermost = True

        for i in range(len(self.up_modules)):
            if i == 0 and innermost:
                data = self.up_modules[i]((data, stack_down.pop()))
            else:
                data = self.up_modules[i]((data, stack_down.pop()), precomputed_up=self.upsample)

        return data.x

    @property
    def num_features(self):
        return sum(self._input_nc_feats)
