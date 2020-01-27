import logging

from .base import Segmentation_MP
from src.modules.KPConv import *
from src.models.base_model import BaseModel
from src.models.base_architectures.unet import UnwrappedUnetBasedModel

log = logging.getLogger(__name__)


class KPConvPaper(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        feature_down = self._assemble_down_convs(option)
        self._assemble_up_convs(feature_down, option)

        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)

        last_mlp_opt = option.mlp_cls

        self.FC_layer = pt_utils.Seq(last_mlp_opt.nn[0] + self._num_categories)
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.conv1d(last_mlp_opt.nn[i], bn=True, bias=False)
        if last_mlp_opt.dropout:
            self.FC_layer.dropout(p=last_mlp_opt.dropout)

        self.FC_layer.conv1d(self._num_classes, activation=None)
        self.loss_names = ["loss_seg"]

    @staticmethod
    def _assemble_down_convs(option):
        subsampling = option.first_subsampling
        initial_feature_size = option.initial_feature_size
        block_names = [
            [SimpleBlock, ResnetBBlock],
            [ResnetBBlock, ResnetBBlock],
            [ResnetBBlock, ResnetBBlock],
            [ResnetBBlock, ResnetBBlock],
            [ResnetBBlock, ResnetBBlock],
        ]
        block_params = [
            [
                {"down_conv_nn": [1, initial_feature_size], "grid_size": subsampling, "is_strided": False},
                {
                    "down_conv_nn": [initial_feature_size, initial_feature_size],
                    "grid_size": subsampling,
                    "is_strided": False,
                },
            ]
        ]
        feature_sizes = [initial_feature_size]
        for i in range(1, len(block_names)):
            subsampling *= 2
            feature_sizes.append(2 * feature_sizes[-1])
            new_params = [
                {"down_conv_nn": [feature_sizes[-2], feature_sizes[-1]], "grid_size": subsampling, "is_strided": True,},
                {
                    "down_conv_nn": [feature_sizes[-1], feature_sizes[-1]],
                    "grid_size": subsampling,
                    "is_strided": False,
                },
            ]
            block_params.append(new_params)
        setattr(option.down_conv, "block_names", block_names)
        setattr(option.down_conv, "block_params", block_params)
        return feature_sizes

    @staticmethod
    def _assemble_up_conv(down_features, option):
        up_conv_nn = []
        for i in range(1, down_features):
            up_conv_nn.append([down_features[-i] + down_features[-i - 1], down_features[-i - 1]])
        setattr(option, "up_conv_nn", up_conv_nn)

    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input = data
        self.labels = data.y

    def forward(self) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        pass

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results

        self.loss_seg = F.nll_loss(self.output, self.labels) + self.get_internal_loss()

        if torch.isnan(self.loss_seg):
            import pdb

            pdb.set_trace()
        self.loss_seg.backward()  # calculate gradients of network G w.r.t. loss_G


class KPConvSeg(Segmentation_MP):
    """ Basic implementation of KPConv"""

    def set_input(self, data):
        self.input = data
        self.batch_idx = data.batch
        self.labels = data.y
