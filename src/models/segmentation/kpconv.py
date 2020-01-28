import logging
import queue
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.nn import Sequential, BatchNorm1d, Dropout, Linear

from .base import Segmentation_MP
from src.modules.KPConv import *
from src.models.base_model import BaseModel
from src.models.base_architectures.unet import UnwrappedUnetBasedModel

log = logging.getLogger(__name__)


class KPConvPaper(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # Extract parameters from the dataset
        self._num_classes = dataset.num_classes
        self._weight_classes = dataset.weight_classes
        self._use_category = option.use_category
        if self._use_category:
            if not dataset.class_to_segments:
                raise ValueError(
                    "The dataset needs to specify a class_to_segments property when using category information for segmentation"
                )
            self._num_categories = len(dataset.class_to_segments.keys())
            log.info("Using category information for the predictions with %i categories", self._num_categories)
        else:
            self._num_categories = 0

        # ASsemble encoder / decoder
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)

        # Build final MLP
        last_mlp_opt = option.mlp_cls
        in_feat = last_mlp_opt.nn[0] + self._num_categories
        self.FC_layer = Sequential()
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.add_module(
                str(i),
                Sequential(
                    *[Linear(in_feat, last_mlp_opt.nn[i], bias=False), LeakyReLU(0.2), BatchNorm1d(last_mlp_opt.nn[i])]
                ),
            )
            in_feat = last_mlp_opt.nn[i]

        if last_mlp_opt.dropout:
            self.FC_layer.add_module("Dropout", Dropout(p=last_mlp_opt.dropout))

        self.FC_layer.add_module("Class", Lin(in_feat, self._num_classes, bias=False))
        self.loss_names = ["loss_seg"]

    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        if hasattr(data, "x"):
            data.x = torch.cat([torch.ones(data.x.shape[0], dtype=torch.float).unsqueeze(-1), data.x], dim=-1)
        self.input = data
        self.labels = data.y

    def forward(self) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        stack_down = []

        data = self.input
        for i in range(len(self.down_modules) - 1):
            data = self.down_modules[i](data)
            stack_down.append(data)

        data = self.down_modules[-1](data)
        for i in range(len(self.up_modules)):
            data = self.up_modules[i]((data, stack_down.pop()))

        last_feature = data.x
        if self._use_category:
            num_points = data.pos.shape[1]
            cat_one_hot = (
                torch.zeros((data.pos.shape[0], self._num_categories, num_points)).float().to(self.category.device)
            )
            cat_one_hot.scatter_(1, self.category.repeat(1, num_points).unsqueeze(1), 1)
            last_feature = torch.cat((last_feature, cat_one_hot), dim=1)

        self.output = self.FC_layer(last_feature)
        return self.output

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
