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

SPECIAL_NAMES = ['radius']


class UnetBasedModel(nn.Module):
    """Create a Unet-based generator"""

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

    def fetch_arguments_up_and_down(self, opt, index, count_convs):
        # Defines down arguments
        args_down = self.fetch_arguments_from_list(opt.down_conv, index)
        args_down['down_conv_cls'] = self.down_conv_cls

        # Defines up arguments
        args_up = self.fetch_arguments_from_list(opt.up_conv, count_convs - index)
        args_up['up_conv_cls'] = self.up_conv_cls
        return args_up, args_down

    def __init__(self, opt, num_classes, modules_lib):
        """Construct a Unet generator
        Parameters:
            opt - options for the network generation
            num_class - output of the network
            modules_lib - all modules that can be used in the UNet
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetBasedModel, self).__init__()

        num_convs = len(opt.down_conv.down_conv_nn)

        self.down_conv_cls = getattr(modules_lib, opt.down_conv.module_name, None)
        self.up_conv_cls = getattr(modules_lib, opt.up_conv.module_name, None)

        # construct unet structure
        contains_global = hasattr(opt, "innermost")
        if contains_global:
            assert len(opt.down_conv.down_conv_nn) + 1 == len(opt.up_conv.up_conv_nn)
            args_up = self.fetch_arguments_from_list(opt.up_conv, 0)
            args_up['up_conv_cls'] = self.up_conv_cls
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
            inner_module = [inner_module_cls(**args_innermost)]
            self.inner = nn.Sequential(*inner_module)
            upconv_cls = self.get_from_kwargs(args_up, 'up_conv_cls')
            up = [upconv_cls(**args_up)]
            self.up = nn.Sequential(*up)
        else:
            downconv_cls = self.get_from_kwargs(args_down, 'down_conv_cls')
            upconv_cls = self.get_from_kwargs(args_up, 'up_conv_cls')

            downconv = downconv_cls(**args_down)
            upconv = upconv_cls(**args_up)

            down = [downconv]
            up = [upconv]
            submodule = [submodule]

            self.down = nn.Sequential(*down)
            self.up = nn.Sequential(*up)
            self.submodule = nn.Sequential(*submodule)

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


def MLP(channels, activation=ReLU()):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), activation, BN(channels[i]))
        for i in range(1, len(channels))
    ])
