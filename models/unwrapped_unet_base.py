import torch
from torch import nn
from abc import abstractmethod
import torch_geometric
from torch_geometric.nn import (
    global_max_pool,
    global_mean_pool,
    fps,
    radius,
    knn_interpolate,
)
from torch.nn import (
    Sequential as Seq,
    Linear as Lin,
    ReLU,
    LeakyReLU,
    BatchNorm1d as BN,
    Dropout,
)
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from collections import defaultdict
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
import logging

from datasets.base_dataset import BaseDataset
from .base_model import BaseModel, SPECIAL_NAMES, BaseFactory
from .utils import is_omegaconf_list
from .core_modules import Identity

log = logging.getLogger(__name__)


class UnwrappedUnetBasedModel(BaseModel):
    """Create a Unet unwrapped generator"""

    def _save_sampling_and_search(self, down_conv, index):
        self._sampling_and_search_dict[index] = [
            getattr(down_conv, "sampler", None),
            getattr(down_conv, "neighbour_finder", None),
        ]

    def __init__(self, opt, model_type, dataset: BaseDataset, modules_lib):
        """Construct a Unet unwrapped generator

        The layers will be appended within lists with the following names
        * down_modules : Contains all the down module
        * inner_modules : Contain one or more inner modules
        * up_modules: Contains all the up module

        Parameters:
            opt - options for the network generation
            model_type - type of the model to be generated
            num_class - output of the network
            modules_lib - all modules that can be used in the UNet

        For a recursive implementation. See UnetBaseModel.

        opt is expected to contains the following keys:
        * down_conv
        * up_conv
        * OPTIONAL: innermost

        """
        super(UnwrappedUnetBasedModel, self).__init__(opt)
        # detect which options format has been used to define the model

        opt

        if is_omegaconf_list(opt.down_conv) or "down_conv_nn" not in opt.down_conv:
            raise NotImplementedError
        else:
            self._init_from_compact_format(opt, model_type, dataset, modules_lib)

    def _get_from_kwargs(self, kwargs, name):
        module = kwargs[name]
        kwargs.pop(name)
        return module

    def _create_down_and_up_modules(self, args_down, args_up, modules_lib):
        downconv_cls = self._get_from_kwargs(args_down, "down_conv_cls")
        upconv_cls = self._get_from_kwargs(args_up, "up_conv_cls")
        return downconv_cls(**args_down), upconv_cls(**args_up)

    def _create_inner_modules(self, args_innermost, args_up, modules_lib):
        module_name = self._get_from_kwargs(args_innermost, "module_name")
        inner_module_cls = getattr(modules_lib, module_name)
        upconv_cls = self._get_from_kwargs(args_up, "up_conv_cls")
        return inner_module_cls(**args_innermost), upconv_cls(**args_up)

    def _init_from_compact_format(self, opt, model_type, dataset, modules_lib):
        """Create a unetbasedmodel from the compact options format - where the
        same convolution is given for each layer, and arguments are given
        in lists
        """

        self.down_modules = nn.ModuleList()
        self.inner_modules = nn.ModuleList()
        self.up_modules = nn.ModuleList()

        num_convs = len(opt.down_conv.down_conv_nn)

        # Factory for creating up and down modules
        factory_module_cls = self._get_factory(model_type, modules_lib)
        down_conv_cls_name = opt.down_conv.module_name
        up_conv_cls_name = opt.up_conv.module_name
        self._factory_module = factory_module_cls(
            down_conv_cls_name, up_conv_cls_name, modules_lib
        )  # Create the factory object

        # construct unet structure
        contains_global = hasattr(opt, "innermost") and opt.innermost is not None
        if contains_global:
            assert len(opt.down_conv.down_conv_nn) + 1 == len(opt.up_conv.up_conv_nn)

            args_up = self._fetch_arguments_from_list(opt.up_conv, 0)
            args_up["up_conv_cls"] = self._factory_module.get_module(0, "UP")

            inner, up = self._create_inner_modules(opt.innermost, args_up, modules_lib)
            self.inner_modules.append(inner)
            self.up_modules.append(up)

        else:
            self.inner_modules.append(Identity())

        if num_convs > 1:
            for index in range(num_convs - 1, 0, -1):
                args_up, args_down = self._fetch_arguments_up_and_down(opt, index)

                down_module, up_module = self._create_down_and_up_modules(args_down, args_up, modules_lib)
                self.down_modules.append(down_module)
                self.up_modules.append(up_module)
                self._save_sampling_and_search(down_module, index)
        else:
            index = num_convs

        index -= 1
        args_up, args_down = self._fetch_arguments_up_and_down(opt, index)
        args_down["nb_feature"] = dataset.feature_dimension
        args_up["nb_feature"] = dataset.feature_dimension

        down_module, up_module = self._create_down_and_up_modules(args_down, args_up, modules_lib)
        self.down_modules.append(down_module)
        self.up_modules.append(up_module)
        self._save_sampling_and_search(down_module, index)

        self.down_modules = self.down_modules[::-1]

        log.info(self)

    def _get_factory(self, model_name, modules_lib) -> BaseFactory:
        factory_module_cls = getattr(modules_lib, "{}Factory".format(model_name), None)
        if factory_module_cls is None:
            factory_module_cls = BaseFactory
        return factory_module_cls

    def _fetch_arguments_from_list(self, opt, index):
        """Fetch the arguments for a single convolution from multiple lists
        of arguments - for models specified in the compact format.
        """
        args = {}
        for o, v in opt.items():
            name = str(o)
            if is_omegaconf_list(v) and len(getattr(opt, o)) > 0:
                if name[-1] == "s" and name not in SPECIAL_NAMES:
                    name = name[:-1]
                v_index = v[index]
                if is_omegaconf_list(v_index):
                    v_index = list(v_index)
                args[name] = v_index
            else:
                if is_omegaconf_list(v):
                    v = list(v)
                args[name] = v
        args["precompute_multi_scale"] = self._precompute_multi_scale
        return args

    def _fetch_arguments_up_and_down(self, opt, index):
        # Defines down arguments
        args_down = self._fetch_arguments_from_list(opt.down_conv, index)
        args_down["index"] = index
        args_down["down_conv_cls"] = self._factory_module.get_module(index, "DOWN")

        # Defines up arguments
        idx = len(getattr(opt.up_conv, "up_conv_nn")) - index - 1
        args_up = self._fetch_arguments_from_list(opt.up_conv, idx)
        args_up["index"] = index
        args_up["up_conv_cls"] = self._factory_module.get_module(index, "UP")
        return args_up, args_down

    def _flatten_compact_options(self, opt):
        """Converts from a dict of lists, to a list of dicts
        """
        flattenedOpts = []

        for index in range(int(1e6)):
            try:
                flattenedOpts.append(DictConfig(self._fetch_arguments_from_list(opt, index)))
            except IndexError:
                break

        return flattenedOpts
