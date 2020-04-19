from . import ModelFactory
from torch_points3d.core.data_transform import AddOnes
from torch_points3d.modules.KPConv import *
from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.modules.KPConv import *
from torch_points3d.core.base_conv.partial_dense import *
from torch_points3d.datasets.multiscale_data import MultiScaleBatch
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel


class KPConv(ModelFactory, UnwrappedUnetBasedModel):

    MODULE_NAME = "kpconv"
    CONV_TYPE = "partial_dense"

    _transforms = [AddOnes()]
    _list_add_to_x = [True]
    _feat_names = ["ones"]
    _input_nc_feats = [1]
    _delete_feats = [True]

    def __init__(self, *args, **kwargs):
        super(KPConv, self).__init__(*args, **kwargs)

        self._build()

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
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

    def forward(self) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""

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
