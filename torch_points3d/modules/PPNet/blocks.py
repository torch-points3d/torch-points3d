import torch
import sys
from torch.nn import Linear as Lin

from .ops import PosPoolLayer
from torch_points3d.core.common_modules.base_modules import BaseModule, FastBatchNorm1d
from torch_points3d.core.spatial_ops import RadiusNeighbourFinder
from torch_points3d.core.data_transform import GridSampling3D
from torch_points3d.utils.enums import ConvolutionFormat
from torch_points3d.utils.config import is_list


class SimpleBlock(BaseModule):
    """
    simple layer with PosPool
    we can perform a stride version (just change the query and the neighbors)
    """
    CONV_TYPE = ConvolutionFormat.PARTIAL_DENSE.value
    DENSITY_PARAMETER = 2.5

    def __init__(
            self,
            down_conv_nn=None,
            grid_size=None,
            prev_grid_size=None,
            sigma=1.0,
            max_num_neighbors=16,
            position_embedding='xyz',
            reduction='avg',
            output_conv=False,
            activation=torch.nn.LeakyReLU(negative_slope=0.2),
            bn_momentum=0.01,
            bn=FastBatchNorm1d,
            **kwargs,
    ):
        super(SimpleBlock, self).__init__()
        assert len(down_conv_nn) == 2
        num_inputs, num_outputs = down_conv_nn

        search_radius = self.DENSITY_PARAMETER * sigma * prev_grid_size
        self.neighbour_finder = RadiusNeighbourFinder(search_radius, max_num_neighbors, conv_type=self.CONV_TYPE)

        self.pospool = PosPoolLayer(num_inputs,
                                    num_outputs,
                                    search_radius,
                                    position_embedding=position_embedding,
                                    reduction=reduction,
                                    output_conv=output_conv,
                                    activation=activation,
                                    bn_momentum=bn_momentum,
                                    bn=bn)

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

        x = self.pospool(q_pos, data.pos, idx_neighboors, data.x)

        query_data.x = x
        query_data.block_idx = data.block_idx + 1
        return query_data

    def extra_repr(self):
        return "Nb parameters: {}; {}; {}".format(self.nb_params, self.sampler, self.neighbour_finder)


class SimpleInputBlock(BaseModule):
    """
    a 1x1 conv and a simple layer with PosPool for input data
    we can perform a stride version (just change the query and the neighbors)
    """
    CONV_TYPE = ConvolutionFormat.PARTIAL_DENSE.value
    DENSITY_PARAMETER = 2.5

    def __init__(
            self,
            down_conv_nn=None,
            grid_size=None,
            prev_grid_size=None,
            sigma=1.0,
            max_num_neighbors=16,
            position_embedding='xyz',
            reduction='avg',
            output_conv=False,
            activation=torch.nn.LeakyReLU(negative_slope=0.2),
            bn_momentum=0.01,
            bn=FastBatchNorm1d,
            **kwargs,
    ):
        super(SimpleInputBlock, self).__init__()
        assert len(down_conv_nn) == 3
        num_inputs, d_2, num_outputs = down_conv_nn

        if bn:
            self.unary_1 = torch.nn.Sequential(
                Lin(num_inputs, d_2, bias=False), bn(d_2, momentum=bn_momentum), activation
            )
        else:
            self.unary_1 = torch.nn.Sequential(Lin(num_inputs, d_2, bias=False), activation)

        search_radius = self.DENSITY_PARAMETER * sigma * prev_grid_size
        self.neighbour_finder = RadiusNeighbourFinder(search_radius, max_num_neighbors, conv_type=self.CONV_TYPE)

        self.pospool = PosPoolLayer(d_2,
                                    num_outputs,
                                    search_radius,
                                    position_embedding=position_embedding,
                                    reduction=reduction,
                                    output_conv=output_conv,
                                    activation=activation,
                                    bn_momentum=bn_momentum,
                                    bn=bn)

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

        x = self.unary_1(data.x)
        x = self.pospool(q_pos, data.pos, idx_neighboors, x)

        query_data.x = x
        query_data.block_idx = data.block_idx + 1
        return query_data

    def extra_repr(self):
        return "Nb parameters: {}; {}; {}".format(self.nb_params, self.sampler, self.neighbour_finder)


class ResnetBBlock(BaseModule):
    """ ResNet bottleneck block with PosPool
    Arguments:
        down_conv_nn (len of 2) : sizes of input, output
        grid_size : size of the grid
        prev_grid_size : size of the grid at previous step.
                In case of a strided block, this is different than grid_size
        max_num_neighbors : maximum number of neighboors for the neighboor search
        position_embedding: Position Embedding type
        reduction: Reduction type in local aggregation
        output_conv: whether to use a convolution after aggregation
        activation : activation function
        has_bottleneck: wether to use the bottleneck or not
        bottleneck_ratio: bottleneck ratio, intermediate =  num_outputs // bottleneck ratio
        bn_momentum: the value used for the running_mean and running_var
        bn : batch norm (can be None -> no batch norm
    """

    CONV_TYPE = ConvolutionFormat.PARTIAL_DENSE.value

    def __init__(
            self,
            down_conv_nn=None,
            grid_size=None,
            prev_grid_size=None,
            sigma=1,
            max_num_neighbors=16,
            position_embedding='xyz',
            reduction='avg',
            output_conv=False,
            activation=torch.nn.LeakyReLU(negative_slope=0.2),
            has_bottleneck=True,
            bottleneck_ratio=2,
            bn_momentum=0.01,
            bn=FastBatchNorm1d,
            **kwargs,
    ):
        super(ResnetBBlock, self).__init__()
        assert len(down_conv_nn) == 2, "down_conv_nn should be of size 2"
        num_inputs, num_outputs = down_conv_nn
        d_2 = num_outputs // bottleneck_ratio

        self.is_strided = prev_grid_size != grid_size
        self.has_bottleneck = has_bottleneck

        # Main branch
        if self.has_bottleneck:
            channel_size = [d_2, d_2]
        else:
            channel_size = [num_inputs, num_outputs]

        self.aggregation = SimpleBlock(
            down_conv_nn=channel_size,
            grid_size=grid_size,
            prev_grid_size=prev_grid_size,
            sigma=sigma,
            max_num_neighbors=max_num_neighbors,
            position_embedding=position_embedding,
            reduction=reduction,
            output_conv=output_conv,
            activation=activation,
            bn_momentum=bn_momentum,
            bn=bn)

        if self.has_bottleneck:
            if bn:
                self.unary_1 = torch.nn.Sequential(
                    Lin(num_inputs, d_2, bias=False), bn(d_2, momentum=bn_momentum), activation
                )
                self.unary_2 = torch.nn.Sequential(
                    Lin(d_2, num_outputs, bias=False), bn(num_outputs, momentum=bn_momentum)
                )
            else:
                self.unary_1 = torch.nn.Sequential(Lin(num_inputs, d_2, bias=False), activation)
                self.unary_2 = torch.nn.Sequential(Lin(d_2, num_outputs, bias=False))

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
        output = self.aggregation(output, precomputed=precomputed)
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
        output.x = self.activation(output.x)
        return output

    @property
    def sampler(self):
        return self.aggregation.sampler

    @property
    def neighbour_finder(self):
        return self.aggregation.neighbour_finder

    def extra_repr(self):
        return "Nb parameters: %i" % self.nb_params


class PPStageBlock(BaseModule):
    """ PPNet Stage block (usually strided + non strided)

    Arguments: Accepted kwargs
        block_names: Name of the blocks to be used as part of this dual block
        down_conv_nn: Size of the convs e.g. [64,128],
        grid_size: Size of the grid for each block,
        prev_grid_size: Size of the grid in the previous KPConv
        has_bottleneck: Wether a block should implement the bottleneck
        bottleneck_ratio: bottleneck ratio, intermediate =  num_outputs // bottleneck ratio
        max_num_neighbors: Max number of neighboors for the radius search,
        position_embedding: Position Embedding type
        reduction: Reduction type in local aggregation
        output_conv: whether to use a convolution after aggregation
        bn_momentum: the value used for the running_mean and running_var

    """

    def __init__(
            self,
            block_names=None,
            down_conv_nn=None,
            grid_size=None,
            prev_grid_size=None,
            has_bottleneck=None,
            bottleneck_ratio=None,
            max_num_neighbors=None,
            position_embedding=None,
            reduction=None,
            output_conv=None,
            bn_momentum=None,
            **kwargs,
    ):
        super(PPStageBlock, self).__init__()

        assert len(block_names) == len(down_conv_nn)

        self.blocks = torch.nn.ModuleList()
        for i, class_name in enumerate(block_names):
            # Constructing extra keyword arguments
            block_kwargs = {}
            for key, arg in kwargs.items():
                block_kwargs[key] = arg[i] if is_list(arg) else arg

            # Building the block
            aggcls = getattr(sys.modules[__name__], class_name)
            block = aggcls(
                down_conv_nn=down_conv_nn[i],
                grid_size=grid_size[i],
                prev_grid_size=prev_grid_size[i],
                has_bottleneck=has_bottleneck[i],
                max_num_neighbors=max_num_neighbors[i],
                bottleneck_ratio=bottleneck_ratio,
                position_embedding=position_embedding,
                reduction=reduction,
                output_conv=output_conv,
                bn_momentum=bn_momentum,
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
