from torch import nn
from torch_geometric.nn import (
    global_max_pool,
    global_mean_pool,
    fps,
    radius,
    knn_interpolate,
)
from torch.nn import (
    Linear as Lin,
    ReLU,
    LeakyReLU,
    BatchNorm1d as BN,
    Dropout,
)
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
import logging

from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.models.base_model import BaseModel
from torch_points3d.models.base_architectures import BaseFactory
from torch_points3d.core.common_modules.base_modules import Identity
from torch_points3d.core.losses import instantiate_loss_or_miner
from torch_points3d.utils.config import is_list

log = logging.getLogger(__name__)

SPECIAL_NAMES = ["radius", "max_num_neighbors", "block_names"]


############################# Backbone Base ###################################


class BackboneBasedModel(BaseModel):
    """
    create a backbone-based generator:
    This is simply an encoder
    (can be used in classification, regression, metric learning and so one)
    """

    def _save_sampling_and_search(self, down_conv):
        sampler = getattr(down_conv, "sampler", None)
        if is_list(sampler):
            self._spatial_ops_dict["sampler"] = sampler + self._spatial_ops_dict["sampler"]
        else:
            self._spatial_ops_dict["sampler"] = [sampler] + self._spatial_ops_dict["sampler"]

        neighbour_finder = getattr(down_conv, "neighbour_finder", None)
        if is_list(neighbour_finder):
            self._spatial_ops_dict["neighbour_finder"] = neighbour_finder + self._spatial_ops_dict["neighbour_finder"]
        else:
            self._spatial_ops_dict["neighbour_finder"] = [neighbour_finder] + self._spatial_ops_dict["neighbour_finder"]

    def __init__(self, opt, model_type, dataset: BaseDataset, modules_lib):

        """Construct a backbone generator (It is a simple down module)
        Parameters:
            opt - options for the network generation
            model_type - type of the model to be generated
            modules_lib - all modules that can be used in the backbone


        opt is expected to contains the following keys:
        * down_conv
        """

        super(BackboneBasedModel, self).__init__(opt)
        self._spatial_ops_dict = {"neighbour_finder": [], "sampler": []}

        # detect which options format has been used to define the model
        if is_list(opt.down_conv) or "down_conv_nn" not in opt.down_conv:
            raise NotImplementedError
        else:
            self._init_from_compact_format(opt, model_type, dataset, modules_lib)

    def _get_from_kwargs(self, kwargs, name):
        module = kwargs[name]
        kwargs.pop(name)
        return module

    def _init_from_compact_format(self, opt, model_type, dataset, modules_lib):
        """Create a backbonebasedmodel from the compact options format - where the
        same convolution is given for each layer, and arguments are given
        in lists
        """
        num_convs = len(opt.down_conv.down_conv_nn)
        self.down_modules = nn.ModuleList()
        factory_module_cls = self._get_factory(model_type, modules_lib)
        down_conv_cls_name = opt.down_conv.module_name
        self._factory_module = factory_module_cls(down_conv_cls_name, None, modules_lib)
        # Down modules
        for i in range(num_convs):
            args = self._fetch_arguments(opt.down_conv, i, "DOWN")
            conv_cls = self._get_from_kwargs(args, "conv_cls")
            down_module = conv_cls(**args)
            self._save_sampling_and_search(down_module)
            self.down_modules.append(down_module)

        self.metric_loss_module, self.miner_module = BaseModel.get_metric_loss_and_miner(
            getattr(opt, "metric_loss", None), getattr(opt, "miner", None)
        )

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
            if is_list(v) and len(getattr(opt, o)) > 0:
                if name[-1] == "s" and name not in SPECIAL_NAMES:
                    name = name[:-1]
                v_index = v[index]
                if is_list(v_index):
                    v_index = list(v_index)
                args[name] = v_index
            else:
                if is_list(v):
                    v = list(v)
                args[name] = v
        return args

    def _fetch_arguments(self, conv_opt, index, flow="DOWN"):
        """ Fetches arguments for building a convolution down

        Arguments:
            conv_opt
            index in sequential order (as they come in the config)
            flow "DOWN"
        """
        args = self._fetch_arguments_from_list(conv_opt, index)
        args["conv_cls"] = self._factory_module.get_module(flow)
        args["index"] = index
        return args

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
