import torch
import sys

from .kernels import KPConvLayer
from src.core.common_modules.base_modules import UnaryConv
from src.core.neighbourfinder import RadiusNeighbourFinder
from src.core.data_transform import GridSampling
from src.utils.enums import ConvolutionFormat


class SimpleBlock(torch.nn.Module):
    """
    simple layer with KPConv convolution -> activation -> BN
    we can perform a stride version (just change the query and the neighbors)
    """

    CONV_TYPE = ConvolutionFormat.PARTIAL_DENSE.value[-1]

    def __init__(
        self,
        down_conv_nn=None,
        grid_size=None,
        is_strided=False,
        sigma=1.0,
        density_parameter=2.5,
        max_num_neighbors=16,
        activation=torch.nn.LeakyReLU(negative_slope=0.2),
        bn_momentum=0.1,
        bn=torch.nn.BatchNorm1d,
        **kwargs
    ):
        super(SimpleBlock, self).__init__()
        assert len(down_conv_nn) == 2
        num_inputs, num_outputs = down_conv_nn
        self.grid_size = grid_size
        radius = density_parameter * sigma * grid_size
        self.kp_conv = KPConvLayer(num_inputs, num_outputs, point_influence=grid_size * sigma)
        if bn:
            self.bn = bn(num_outputs, momentum=bn_momentum)
        else:
            self.bn = None
        self.activation = activation

        self.neighbour_finder = RadiusNeighbourFinder(radius, max_num_neighbors, conv_type=self.CONV_TYPE)
        if is_strided:
            self.sampler = GridSampling(grid_size)
        else:
            self.sampler = None

    def forward(self, data):
        if self.sampler:
            querry_data = self.sampler(data.clone())
        else:
            querry_data = data

        q_pos, q_batch = querry_data.pos, querry_data.batch
        # if hasattr(data, "idx_neighboors") and data.idx_neighboors.shape[0] == q_pos.shape[0]:
        #     idx_neighboors = data.idx_neighboors
        # else:
        idx_neighboors, _ = self.neighbour_finder(data.pos, q_pos, batch_x=data.batch, batch_y=q_batch)
        x = self.kp_conv(q_pos, data.pos, idx_neighboors, data.x,)
        if self.bn:
            x = self.bn(x)
        x = self.activation(x)

        querry_data.x = x
        querry_data.idx_neighboors = idx_neighboors
        return querry_data

    def extra_repr(self):
        return str(self.sampler) + "," + str(self.neighbour_finder)


class ResnetBBlock(torch.nn.Module):
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
        grid_size : size of the grid in case of a strided block,
    """

    CONV_TYPE = ConvolutionFormat.PARTIAL_DENSE.value[-1]

    def __init__(
        self,
        down_conv_nn=None,
        grid_size=None,
        is_strided=False,
        sigma=1,
        density_parameter=2.5,
        max_num_neighbors=16,
        activation=torch.nn.LeakyReLU(negative_slope=0.2),
        has_bottleneck=True,
        bn_momentum=0.1,
        bn=torch.nn.BatchNorm1d,
        **kwargs
    ):
        super(ResnetBBlock, self).__init__()
        assert len(down_conv_nn) == 2 or len(down_conv_nn) == 3, "down_conv_nn should be of size 2 or 3"
        if len(down_conv_nn) == 2:
            num_inputs, num_outputs = down_conv_nn
            d_2 = num_outputs // 4
        else:
            num_inputs, d_2, num_outputs = down_conv_nn
        self.is_strided = is_strided
        self.grid_size = grid_size
        self.has_bottleneck = has_bottleneck
        if self.is_strided:
            self.sampler = GridSampling(grid_size)

        # Main branch
        if self.has_bottleneck:
            kp_size = [d_2, d_2]
        else:
            kp_size = [num_inputs, num_outputs]

        self.kp_conv = SimpleBlock(
            down_conv_nn=kp_size,
            grid_size=grid_size,
            is_strided=is_strided,
            sigma=sigma,
            density_parameter=density_parameter,
            max_num_neighbors=max_num_neighbors,
            activation=activation,
            bn_momentum=bn_momentum,
            bn=bn,
        )

        if self.has_bottleneck:
            if bn:
                self.unary_1 = torch.nn.Sequential(
                    UnaryConv(num_inputs, d_2), bn(d_2, momentum=bn_momentum), activation
                )
                self.unary_2 = torch.nn.Sequential(
                    UnaryConv(d_2, num_outputs), bn(num_outputs, momentum=bn_momentum), activation
                )
            else:
                self.unary_1 = torch.nn.Sequential(UnaryConv(num_inputs, d_2), activation)
                self.unary_2 = torch.nn.Sequential(UnaryConv(d_2, num_outputs), activation)

        # Shortcut
        if num_inputs != num_outputs:
            if bn:
                self.shortcut_op = UnaryConv(num_inputs, num_outputs)
            else:
                self.shortcut_op = torch.nn.Sequential(
                    UnaryConv(num_inputs, num_outputs), bn(num_outputs, momentum=bn_momentum)
                )
        else:
            self.shortcut_op = torch.nn.Identity()

        # Final activation
        self.activation = activation

    def forward(self, data):
        """
            data: x, pos, batch_idx and idx_neighbour when the neighboors of each point in pos have already been computed
        """
        # Main branch
        output = data.clone()
        shortcut_x = data.x
        if self.has_bottleneck:
            output.x = self.unary_1(output.x)
        output = self.kp_conv(output)
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


class KPDualBlock(torch.nn.Module):
    def __init__(
        self, block_names=None, down_conv_nn=None, grid_size=None, is_strided=None, has_bottleneck=None, **kwargs
    ):
        super(KPDualBlock, self).__init__()

        assert len(block_names) == len(down_conv_nn)
        self.blocks = torch.nn.ModuleList()
        for i, class_name in enumerate(block_names):
            kpcls = getattr(sys.modules[__name__], class_name)
            block = kpcls(
                down_conv_nn=down_conv_nn[i],
                grid_size=grid_size[i],
                is_strided=is_strided[i],
                has_bottleneck=has_bottleneck[i],
            )
            self.blocks.append(block)

    def forward(self, data):
        for block in self.blocks:
            data = block(data)
        return data
