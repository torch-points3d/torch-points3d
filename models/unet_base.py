import torch
from torch import nn
from abc import abstractmethod
import torch_geometric
from torch_geometric.nn import global_max_pool, global_mean_pool, fps, radius, knn_interpolate
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LeakyReLU, BatchNorm1d as BN, Dropout
from omegaconf.listconfig import ListConfig
from collections import defaultdict
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from .base_model import BaseModel

SPECIAL_NAMES = ['radius']


class UnetBasedModel(BaseModel):
    """Create a Unet-based generator"""

    def __init__(self, opt, model_name, num_classes, modules_lib):
        """Construct a Unet generator
        Parameters:
            opt - options for the network generation
            num_class - output of the network
            modules_lib - all modules that can be used in the UNet
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetBasedModel, self).__init__(opt)

        num_convs = len(opt.down_conv.down_conv_nn)

        self.factory_module_cls, self.has_factory = self.check_if_contains_factory(model_name, modules_lib)

        if self.has_factory:
            self.down_conv_cls_name = opt.down_conv.module_name
            self.up_conv_cls_name = opt.up_conv.module_name
            self.factory_module = self.factory_module_cls(self.down_conv_cls_name, self.up_conv_cls_name, modules_lib) # Create the factory object
        else:
            self.down_conv_cls = getattr(modules_lib, opt.down_conv.module_name, None)
            self.up_conv_cls = getattr(modules_lib, opt.up_conv.module_name, None)

        # construct unet structure
        contains_global = hasattr(opt, "innermost")
        if contains_global:
            assert len(opt.down_conv.down_conv_nn) + 1 == len(opt.up_conv.up_conv_nn)
            
            args_up = self.fetch_arguments_from_list(opt.up_conv, 0)
            args_up = self.get_module_cls(args_up, 0, 'up_conv_cls', "UP")
            
            unet_block = UnetSkipConnectionBlock(args_up=args_up, args_innermost=opt.innermost, modules_lib=modules_lib,
                                                 input_nc=None, submodule=None, norm_layer=None, innermost=True)  # add the innermost layer
        else:
            unet_block = []

        if num_convs > 1:
            for index in range(num_convs - 1, 0, -1):
                args_up, args_down = self.fetch_arguments_up_and_down(opt, index, num_convs)
                unet_block = UnetSkipConnectionBlock(
                    args_up=args_up, args_down=args_down, input_nc=None, submodule=unet_block, norm_layer=None)
        else:
            index = num_convs

        index -= 1
        args_up, args_down = self.fetch_arguments_up_and_down(opt, index, num_convs)
        
        self.model = UnetSkipConnectionBlock(args_up=args_up, args_down=args_down, output_nc=num_classes, input_nc=None, submodule=unet_block,
                                             outermost=True, norm_layer=None)  # add the outermost layer
        print(self)

    def check_if_contains_factory(self, model_name, modules_lib):
        factory_module_cls = getattr(modules_lib, "{}Factory".format(model_name), None)
        if factory_module_cls is None:
            return factory_module_cls, False
        else:
            return factory_module_cls, True

    def fetch_arguments_from_list(self, opt, index):
        args = {}
        for o, v in opt.items():
            name = str(o)
            if (isinstance(getattr(opt, o), ListConfig) and len(getattr(opt, o)) > 0):
                if name[-1] == 's' and name not in SPECIAL_NAMES:
                    name = name[:-1]
                v_index = v[index]
                if isinstance(v_index, ListConfig):
                    v_index = list(v_index)
                args[name] = v_index
            else:
                if isinstance(v, ListConfig):
                    v = list(v)
                args[name] = v
        args['index'] = index
        return args

    def get_module_cls(self, args, index, name, flow):
        if self.has_factory:
            args[name] = self.factory_module.get_module_from_index(index, flow=flow)
        else:
            args[name] = getattr(self, name, None)      
        return args  

    def fetch_arguments_up_and_down(self, opt, index, count_convs):
        # Defines down arguments
        args_down = self.fetch_arguments_from_list(opt.down_conv, index)
        args_down = self.get_module_cls(args_down, index, 'down_conv_cls', "DOWN")

        # Defines up arguments
        args_up = self.fetch_arguments_from_list(opt.up_conv, count_convs - index)
        args_up = self.get_module_cls(args_up, count_convs - index, 'up_conv_cls', "UP")
        return args_up, args_down


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|

    """

    def get_from_kwargs(self, kwargs, name):
        module = kwargs[name]
        kwargs.pop(name)
        return module

    def __init__(self, args_up=None, args_down=None, args_innermost=None, modules_lib=None, submodule=None, outermost=False, innermost=False, use_dropout=False, name=None, *args, **kwargs):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()

        self.outermost = outermost
        self.innermost = innermost

        if innermost:
            assert outermost == False
            module_name = self.get_from_kwargs(args_innermost, 'module_name')
            inner_module_cls = getattr(modules_lib, module_name)
            self.inner = inner_module_cls(**args_innermost)
            upconv_cls = self.get_from_kwargs(args_up, 'up_conv_cls')
            self.up = upconv_cls(**args_up)
        else:
            downconv_cls = self.get_from_kwargs(args_down, 'down_conv_cls')
            upconv_cls = self.get_from_kwargs(args_up, 'up_conv_cls')

            downconv = downconv_cls(**args_down)
            upconv = upconv_cls(**args_up)

            self.down = downconv
            self.submodule = submodule
            self.up = upconv

    def forward(self, data):
        if self.innermost:
            data_out = self.inner(data)
            data = (*data_out, *data)
            return self.up(data)
        else:
            data_out = self.down(data)
            data_out2 = self.submodule(data_out)
            data = (*data_out2, *data)
            return self.up(data)
