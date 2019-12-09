import torch
from torch import nn
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

    def fetch_arguments(self, convs, index):
        keys = convs.keys()
        arguments = {c:getattr(convs, c) for c in keys if (isinstance(getattr(convs, c), ListConfig) and len(getattr(convs, c)) > 0)}
        args = defaultdict()
        for arg_name, arg in arguments.items():
            name = str(arg_name)
            if name[-1] == 's' and name not in SPECIAL_NAMES: name = name[:-1]
            args[name] = arg[index]
        # Add convolution class
        args['down_conv_cls'] = self.down_conv_cls
        args['up_conv_cls'] = self.up_conv_cls
        return args

    def __init__(self, opt, num_classes):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetBasedModel, self).__init__()
        
        num_convs = len(opt.convs.down_conv_nn)
        count_convs = num_convs - 1

        # construct unet structure
        contains_global = hasattr(opt, "global")
        if contains_global:
            unet_block = UnetSkipConnectionBlock(**opt['global'], input_nc=None, submodule=None, norm_layer=None, innermost=True)  # add the innermost layer
        else:
            unet_block = []

        if num_convs > 1:
            for _ in range(num_convs - 1):
                unet_args = self.fetch_arguments(opt.convs, count_convs)
                unet_block = UnetSkipConnectionBlock(**unet_args, input_nc=None, submodule=unet_block, norm_layer=None)
                count_convs -= 1
        unet_args = self.fetch_arguments(opt.convs, count_convs)
        self.model = UnetSkipConnectionBlock(**unet_args, output_nc=num_classes, input_nc=None, submodule=unet_block, \
                    outermost=True, norm_layer=None, name=self._name)  # add the outermost layer
        
        self.upconv = nn.Sequential(*[self.up_conv_cls(up_k=3, up_conv_nn=opt.convs.final_up_conv_nn)])


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|

    """
    def get_from_kwargs(self, kwargs, name):
        module = kwargs[name]
        kwargs.pop(name)
        return module

    def __init__(self, submodule=None, outermost=False, innermost=False, use_dropout=False, name=None, *args, **kwargs):
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
        if name is not None: 
            self._name = name

        if innermost:
            assert outermost == False
            model = [GlobalBaseModule(nn=kwargs.get('nn', None), aggr=kwargs.get('aggr', "max"))]
            self.model = nn.Sequential(*model)
        else:
            downconv_cls = self.get_from_kwargs(kwargs, 'down_conv_cls')
            upconv_cls = self.get_from_kwargs(kwargs, 'up_conv_cls')
            
            downconv = downconv_cls(**kwargs)
            upconv = upconv_cls(**kwargs)

            
            down = [downconv]
            up = [upconv]
            submodule = [submodule]
            
            self.down = nn.Sequential(*down)
            self.up = nn.Sequential(*up)
            self.submodule = nn.Sequential(*submodule)

    def forward(self, data):
        if hasattr(self, "up"):
            data_out = self.down(data)
            data_out2 = self.submodule(data_out)
            data = (*data_out2, *data_out)
            return self.up(data)
        else:
            return self.model(data)


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

class FPModule(torch.nn.Module):
    """ Upsampling module from PointNet++
    
    Arguments:
        k [int] -- number of nearest neighboors used for the interpolation
        up_conv_nn [List[int]] -- list of feature sizes for the uplconv mlp
    
    Returns:
        [type] -- [description]
    """
    def __init__(self, up_k, up_conv_nn, *args, **kwargs):
        super(FPModule, self).__init__()
        self.k = up_k
        self.nn = MLP(up_conv_nn)

    def forward(self, data):
        #print()
        #print([x.shape if x is not None else None for x in data])
        x, pos, batch, x_skip, pos_skip, batch_skip = data
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        data = (x, pos_skip, batch_skip)
        return data

class BaseConvolution(torch.nn.Module):
    def __init__(self, ratio, radius, *args, **kwargs):
        super(BaseConvolution, self).__init__()
        self.ratio = ratio
        self.radius = radius
        self.conv = None # This one should be implemented
        self.max_num_neighbors = kwargs.get("max_num_neighbors", 64)

    def forward(self, data):
        x, pos, batch = data
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.radius, batch, batch[idx],
                          max_num_neighbors=self.max_num_neighbors)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        data = (x, pos, batch)
        return data

class GlobalBaseModule(torch.nn.Module):
    def __init__(self, nn, aggr='max'):
        super(GlobalBaseModule, self).__init__()
        self.nn = MLP(nn)
        self.pool = global_max_pool if aggr == "max" else  global_mean_pool

    def forward(self, data):
        x, pos, batch = data
        #print("GLOBAL", x.shape, pos.shape, batch.shape)
        x = self.nn(torch.cat([x, pos], dim=1))
        #print("GLOBAL", x.shape)
        x = self.pool(x, batch)
        #print("GLOBAL")
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        #print("GLOBAL", x.shape, pos.shape, batch.shape)
        data = (x, pos, batch)
        return data