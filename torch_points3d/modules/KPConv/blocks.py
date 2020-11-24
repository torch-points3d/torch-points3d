import torch
import sys
from torch.nn import Linear as Lin

from .kernels import KPConvLayer, KPConvDeformableLayer
from torch_points3d.core.common_modules.base_modules import BaseModule, FastBatchNorm1d
from torch_points3d.core.spatial_ops import RadiusNeighbourFinder
from torch_points3d.core.data_transform import GridSampling3D
from torch_points3d.utils.enums import ConvolutionFormat
from torch_points3d.core.base_conv.message_passing import GlobalBaseModule
from torch_points3d.core.common_modules.base_modules import Identity
from torch_points3d.utils.config import is_list


class SimpleBlock(BaseModule):
    """
    simple layer with KPConv convolution -> activation -> BN
    we can perform a stride version (just change the query and the neighbors)
    """

    CONV_TYPE = ConvolutionFormat.PARTIAL_DENSE.value
    DEFORMABLE_DENSITY = 5.0
    RIGID_DENSITY = 2.5

    def __init__(
        self,
        down_conv_nn=None,
        grid_size=None,
        prev_grid_size=None,
        sigma=1.0,
        max_num_neighbors=16,
        activation=torch.nn.LeakyReLU(negative_slope=0.1),
        bn_momentum=0.02,
        bn=FastBatchNorm1d,
        deformable=False,
        add_one=False,
        **kwargs,
    ):
        super(SimpleBlock, self).__init__()
        assert len(down_conv_nn) == 2
        num_inputs, num_outputs = down_conv_nn
        if deformable:
            density_parameter = self.DEFORMABLE_DENSITY
            self.kp_conv = KPConvDeformableLayer(
                num_inputs, num_outputs, point_influence=prev_grid_size * sigma, add_one=add_one, **kwargs
            )
        else:
            density_parameter = self.RIGID_DENSITY
            self.kp_conv = KPConvLayer(
                num_inputs, num_outputs, point_influence=prev_grid_size * sigma, add_one=add_one, **kwargs
            )
        search_radius = density_parameter * sigma * prev_grid_size
        self.neighbour_finder = RadiusNeighbourFinder(search_radius, max_num_neighbors, conv_type=self.CONV_TYPE)

        if bn:
            self.bn = bn(num_outputs, momentum=bn_momentum)
        else:
            self.bn = None
        self.activation = activation

        is_strided = prev_grid_size != grid_size
        if is_strided:
            self.sampler = GridSampling3D(grid_size)
        else:
            self.sampler = None

    def forward(self, data, precomputed=None, **kwargs):
        if not hasattr(data, "block_idx"):
            setattr(data, "block_idx", 0)

        if precomputed:
            query_data = precomputed[data.block_idx]
        else:
            if self.sampler:
                query_data = self.sampler(data.clone())
            else:
                query_data = data.clone()

        if precomputed:
            idx_neighboors = query_data.idx_neighboors
            q_pos = query_data.pos
        else:
            q_pos, q_batch = query_data.pos, query_data.batch
            idx_neighboors = self.neighbour_finder(data.pos, q_pos, batch_x=data.batch, batch_y=q_batch)
            query_data.idx_neighboors = idx_neighboors

        x = self.kp_conv(q_pos, data.pos, idx_neighboors, data.x,)
        if self.bn:
            x = self.bn(x)
        x = self.activation(x)

        query_data.x = x
        query_data.block_idx = data.block_idx + 1
        return query_data

    def extra_repr(self):
        return "Nb parameters: {}; {}; {}".format(self.nb_params, self.sampler, self.neighbour_finder)


class ResnetBBlock(BaseModule):
    """ Resnet block with optional bottleneck activated by default
    Arguments:
        down_conv_nn (len of 2 or 3) :
                        sizes of input, intermediate, output.
                        If length == 2 then intermediate =  num_outputs // 4
        radius : radius of the conv kernel
        sigma :
        density_parameter : density parameter for the kernel
        max_num_neighbors : maximum number of neighboors for the neighboor search
        activation : activation function
        has_bottleneck: wether to use the bottleneck or not
        bn_momentum
        bn : batch norm (can be None -> no batch norm)
        grid_size : size of the grid,
        prev_grid_size : size of the grid at previous step.
                        In case of a strided block, this is different than grid_size
    """

    CONV_TYPE = ConvolutionFormat.PARTIAL_DENSE.value

    def __init__(
        self,
        down_conv_nn=None,
        grid_size=None,
        prev_grid_size=None,
        sigma=1,
        max_num_neighbors=16,
        activation=torch.nn.LeakyReLU(negative_slope=0.1),
        has_bottleneck=True,
        bn_momentum=0.02,
        bn=FastBatchNorm1d,
        deformable=False,
        add_one=False,
        **kwargs,
    ):
        super(ResnetBBlock, self).__init__()
        assert len(down_conv_nn) == 2 or len(down_conv_nn) == 3, "down_conv_nn should be of size 2 or 3"
        if len(down_conv_nn) == 2:
            num_inputs, num_outputs = down_conv_nn
            d_2 = num_outputs // 4
        else:
            num_inputs, d_2, num_outputs = down_conv_nn
        self.is_strided = prev_grid_size != grid_size
        self.has_bottleneck = has_bottleneck

        # Main branch
        if self.has_bottleneck:
            kp_size = [d_2, d_2]
        else:
            kp_size = [num_inputs, num_outputs]

        self.kp_conv = SimpleBlock(
            down_conv_nn=kp_size,
            grid_size=grid_size,
            prev_grid_size=prev_grid_size,
            sigma=sigma,
            max_num_neighbors=max_num_neighbors,
            activation=activation,
            bn_momentum=bn_momentum,
            bn=bn,
            deformable=deformable,
            add_one=add_one,
            **kwargs,
        )

        if self.has_bottleneck:
            if bn:
                self.unary_1 = torch.nn.Sequential(
                    Lin(num_inputs, d_2, bias=False), bn(d_2, momentum=bn_momentum), activation
                )
                self.unary_2 = torch.nn.Sequential(
                    Lin(d_2, num_outputs, bias=False), bn(num_outputs, momentum=bn_momentum), activation
                )
            else:
                self.unary_1 = torch.nn.Sequential(Lin(num_inputs, d_2, bias=False), activation)
                self.unary_2 = torch.nn.Sequential(Lin(d_2, num_outputs, bias=False), activation)

        # Shortcut
        if num_inputs != num_outputs:
            if bn:
                self.shortcut_op = torch.nn.Sequential(
                    Lin(num_inputs, num_outputs, bias=False), bn(num_outputs, momentum=bn_momentum)
                )
            else:
                self.shortcut_op = Lin(num_inputs, num_outputs, bias=False)
        else:
            self.shortcut_op = torch.nn.Identity()

        # Final activation
        self.activation = activation

    def forward(self, data, precomputed=None, **kwargs):
        """
            data: x, pos, batch_idx and idx_neighbour when the neighboors of each point in pos have already been computed
        """
        # Main branch
        output = data.clone()
        shortcut_x = data.x
        if self.has_bottleneck:
            output.x = self.unary_1(output.x)
        output = self.kp_conv(output, precomputed=precomputed)
        if self.has_bottleneck:
            output.x = self.unary_2(output.x)

        # Shortcut
        if self.is_strided:
            idx_neighboors = output.idx_neighboors
            shortcut_x = torch.cat([shortcut_x, torch.zeros_like(shortcut_x[:1, :])], axis=0)  # Shadow feature
            neighborhood_features = shortcut_x[idx_neighboors]
            shortcut_x = torch.max(neighborhood_features, dim=1, keepdim=False)[0]

        shortcut = self.shortcut_op(shortcut_x)
        output.x += shortcut
        return output

    @property
    def sampler(self):
        return self.kp_conv.sampler

    @property
    def neighbour_finder(self):
        return self.kp_conv.neighbour_finder

    def extra_repr(self):
        return "Nb parameters: %i" % self.nb_params


class KPDualBlock(BaseModule):
    """ Dual KPConv block (usually strided + non strided)

    Arguments: Accepted kwargs
        block_names: Name of the blocks to be used as part of this dual block
        down_conv_nn: Size of the convs e.g. [64,128],
        grid_size: Size of the grid for each block,
        prev_grid_size: Size of the grid in the previous KPConv
        has_bottleneck: Wether a block should implement the bottleneck
        max_num_neighbors: Max number of neighboors for the radius search,
        deformable: Is deformable,
        add_one: Add one as a feature,
    """

    def __init__(
        self,
        block_names=None,
        down_conv_nn=None,
        grid_size=None,
        prev_grid_size=None,
        has_bottleneck=None,
        max_num_neighbors=None,
        deformable=False,
        add_one=False,
        **kwargs,
    ):
        super(KPDualBlock, self).__init__()

        assert len(block_names) == len(down_conv_nn)
        self.blocks = torch.nn.ModuleList()
        for i, class_name in enumerate(block_names):
            # Constructing extra keyword arguments
            block_kwargs = {}
            for key, arg in kwargs.items():
                block_kwargs[key] = arg[i] if is_list(arg) else arg

            # Building the block
            kpcls = getattr(sys.modules[__name__], class_name)
            block = kpcls(
                down_conv_nn=down_conv_nn[i],
                grid_size=grid_size[i],
                prev_grid_size=prev_grid_size[i],
                has_bottleneck=has_bottleneck[i],
                max_num_neighbors=max_num_neighbors[i],
                deformable=deformable[i] if is_list(deformable) else deformable,
                add_one=add_one[i] if is_list(add_one) else add_one,
                **block_kwargs,
            )
            self.blocks.append(block)

    def forward(self, data, precomputed=None, **kwargs):
        for block in self.blocks:
            data = block(data, precomputed=precomputed)
        return data

    @property
    def sampler(self):
        return [b.sampler for b in self.blocks]

    @property
    def neighbour_finder(self):
        return [b.neighbour_finder for b in self.blocks]

    def extra_repr(self):
        return "Nb parameters: %i" % self.nb_params
