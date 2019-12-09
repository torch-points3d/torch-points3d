import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import global_max_pool, global_mean_pool, fps, radius, knn_interpolate
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LeakyReLU, BatchNorm1d as BN, Dropout
from omegaconf.listconfig import ListConfig
from collections import defaultdict
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.inits import reset

__special__names__ = ['radius']

def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s
class UnetBasedModel(nn.Module):
    """Create a Unet-based generator"""

    def fetch_arguments(self, convs, index):
        keys = convs.keys()
        arguments = {c:getattr(convs, c) for c in keys if (isinstance(getattr(convs, c), ListConfig) and len(getattr(convs, c)) > 0)}
        args = defaultdict()
        for arg_name, arg in arguments.items():
            name = str(arg_name)
            if name[-1] == 's' and name not in __special__names__: name = name[:-1]
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

class PointConv(MessagePassing):
    r"""The PointNet set layer from the `"PointNet: Deep Learning on Point Sets
    for 3D Classification and Segmentation"
    <https://arxiv.org/abs/1612.00593>`_ and `"PointNet++: Deep Hierarchical
    Feature Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ papers

    .. math::
        \mathbf{x}^{\prime}_i = \gamma_{\mathbf{\Theta}} \left( \max_{j \in
        \mathcal{N}(i) \cup \{ i \}} h_{\mathbf{\Theta}} ( \mathbf{x}_j,
        \mathbf{p}_j - \mathbf{p}_i) \right),

    where :math:`\gamma_{\mathbf{\Theta}}` and
    :math:`h_{\mathbf{\Theta}}` denote neural
    networks, *.i.e.* MLPs, and :math:`\mathbf{P} \in \mathbb{R}^{N \times D}`
    defines the position of each point.

    Args:
        local_nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` and
            relative spatial coordinates :obj:`pos_j - pos_i` of shape
            :obj:`[-1, in_channels + num_dimensions]` to shape
            :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        global_nn (torch.nn.Module, optional): A neural network
            :math:`\gamma_{\mathbf{\Theta}}` that maps aggregated node features
            of shape :obj:`[-1, out_channels]` to shape :obj:`[-1,
            final_out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, local_nn=None, global_nn=None, **kwargs):
        super(PointConv, self).__init__(aggr='max')

        self.local_nn = local_nn
        self.global_nn = global_nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.local_nn)
        reset(self.global_nn)

    def forward(self, x, pos, edge_index):
        r"""
        Args:
            x (Tensor): The node feature matrix. Allowed to be :obj:`None`.
            pos (Tensor or tuple): The node position matrix. Either given as
                tensor for use in general message passing or as tuple for use
                in message passing in bipartite graphs.
            edge_index (LongTensor): The edge indices.
        """
        #print(x.shape if x is not None else None, \
        #    [p.shape for p in pos] if isinstance(pos, tuple) else pos.shape)
        if torch.is_tensor(pos):  # Add self-loops for symmetric adjacencies.
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=pos.size(0))

        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_j, pos_i, pos_j):
        msg = pos_j - pos_i
        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

    def update(self, aggr_out):
        if self.global_nn is not None:
            aggr_out = self.global_nn(aggr_out)
        return aggr_out

    def __repr__(self):
        return '{}(local_nn={}, global_nn={})'.format(
            self.__class__.__name__, self.local_nn, self.global_nn)