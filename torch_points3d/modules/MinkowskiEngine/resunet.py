import torch
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from .common import get_norm

from .res16unet import get_block
from .common import NormType


class ResUNet2(ME.MinkowskiNetwork):
    NORM_TYPE = None
    BLOCK_NORM_TYPE = NormType.BATCH_NORM
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 32, 64, 64, 128]

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(
        self, in_channels=3, out_channels=32, bn_momentum=0.01, normalize_feature=True, conv1_kernel_size=5, D=3
    ):
        ME.MinkowskiNetwork.__init__(self, D)
        NORM_TYPE = self.NORM_TYPE
        BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
        CHANNELS = self.CHANNELS
        TR_CHANNELS = self.TR_CHANNELS
        # print(D, in_channels, out_channels, conv1_kernel_size)
        self.normalize_feature = normalize_feature
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=CHANNELS[1],
            kernel_size=conv1_kernel_size,
            stride=1,
            dilation=1,
            has_bias=False,
            dimension=D,
        )
        self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.block1 = get_block(BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            has_bias=False,
            dimension=D,
        )
        self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.block2 = get_block(BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            has_bias=False,
            dimension=D,
        )
        self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.block3 = get_block(BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.conv4 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[3],
            out_channels=CHANNELS[4],
            kernel_size=3,
            stride=2,
            dilation=1,
            has_bias=False,
            dimension=D,
        )
        self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.block4 = get_block(BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.conv4_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[4],
            out_channels=TR_CHANNELS[4],
            kernel_size=3,
            stride=2,
            dilation=1,
            has_bias=False,
            dimension=D,
        )
        self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.block4_tr = get_block(BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.conv3_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[3] + TR_CHANNELS[4],
            out_channels=TR_CHANNELS[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            has_bias=False,
            dimension=D,
        )
        self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.block3_tr = get_block(BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.conv2_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[2] + TR_CHANNELS[3],
            out_channels=TR_CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            has_bias=False,
            dimension=D,
        )
        self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.block2_tr = get_block(BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.conv1_tr = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1] + TR_CHANNELS[2],
            out_channels=TR_CHANNELS[1],
            kernel_size=1,
            stride=1,
            dilation=1,
            has_bias=False,
            dimension=D,
        )

        # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.final = ME.MinkowskiConvolution(
            in_channels=TR_CHANNELS[1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            has_bias=True,
            dimension=D,
        )

    def forward(self, x):
        out_s1 = self.conv1(x)
        out_s1 = self.norm1(out_s1)
        out_s1 = self.block1(out_s1)
        out = MEF.relu(out_s1)

        out_s2 = self.conv2(out)
        out_s2 = self.norm2(out_s2)
        out_s2 = self.block2(out_s2)
        out = MEF.relu(out_s2)

        out_s4 = self.conv3(out)
        out_s4 = self.norm3(out_s4)
        out_s4 = self.block3(out_s4)
        out = MEF.relu(out_s4)

        out_s8 = self.conv4(out)
        out_s8 = self.norm4(out_s8)
        out_s8 = self.block4(out_s8)
        out = MEF.relu(out_s8)

        out = self.conv4_tr(out)
        out = self.norm4_tr(out)
        out = self.block4_tr(out)
        out_s4_tr = MEF.relu(out)

        out = ME.cat(out_s4_tr, out_s4)

        out = self.conv3_tr(out)
        out = self.norm3_tr(out)
        out = self.block3_tr(out)
        out_s2_tr = MEF.relu(out)

        out = ME.cat(out_s2_tr, out_s2)

        out = self.conv2_tr(out)
        out = self.norm2_tr(out)
        out = self.block2_tr(out)
        out_s1_tr = MEF.relu(out)

        out = ME.cat(out_s1_tr, out_s1)
        out = self.conv1_tr(out)
        out = MEF.relu(out)
        out = self.final(out)

        if self.normalize_feature:
            return ME.SparseTensor(
                out.F / torch.norm(out.F, p=2, dim=1, keepdim=True),
                coords_key=out.coords_key,
                coords_manager=out.coords_man,
            )
        else:
            return out


class ResUNetBN2(ResUNet2):
    NORM_TYPE = NormType.BATCH_NORM


class ResUNetBN2B(ResUNet2):
    NORM_TYPE = NormType.BATCH_NORM
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 64, 64]


class ResUNetBN2C(ResUNet2):
    NORM_TYPE = NormType.BATCH_NORM
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 64, 128]


class ResUNetBN2D(ResUNet2):
    NORM_TYPE = NormType.BATCH_NORM
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 128, 128]


class ResUNetBN2E(ResUNet2):
    NORM_TYPE = NormType.BATCH_NORM
    CHANNELS = [None, 128, 128, 128, 256]
    TR_CHANNELS = [None, 64, 128, 128, 128]


class Res2BlockDown(ME.MinkowskiNetwork):

    """
    block for unwrapped Resnet
    """

    def __init__(
        self,
        down_conv_nn,
        kernel_size,
        stride,
        dilation,
        dimension=3,
        bn_momentum=0.01,
        norm_type=NormType.BATCH_NORM,
        block_norm_type=NormType.BATCH_NORM,
        **kwargs
    ):
        ME.MinkowskiNetwork.__init__(self, dimension)
        self.conv = ME.MinkowskiConvolution(
            in_channels=down_conv_nn[0],
            out_channels=down_conv_nn[1],
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            has_bias=False,
            dimension=dimension,
        )
        self.norm = get_norm(norm_type, down_conv_nn[1], bn_momentum=bn_momentum, D=dimension)
        self.block = get_block(block_norm_type, down_conv_nn[1], down_conv_nn[1], bn_momentum=bn_momentum, D=dimension)

    def forward(self, x):

        out_s = self.conv(x)
        out_s = self.norm(out_s)
        out = self.block(out_s)
        return out


class Res2BlockUp(ME.MinkowskiNetwork):

    """
    block for unwrapped Resnet
    """

    def __init__(
        self,
        up_conv_nn,
        kernel_size,
        stride,
        dilation,
        dimension=3,
        bn_momentum=0.01,
        norm_type=NormType.BATCH_NORM,
        block_norm_type=NormType.BATCH_NORM,
        **kwargs
    ):
        ME.MinkowskiNetwork.__init__(self, dimension)
        self.conv = ME.MinkowskiConvolutionTranspose(
            in_channels=up_conv_nn[0],
            out_channels=up_conv_nn[1],
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            has_bias=False,
            dimension=dimension,
        )
        if len(up_conv_nn) == 3:
            self.final = ME.MinkowskiConvolution(
                in_channels=up_conv_nn[1],
                out_channels=up_conv_nn[2],
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                has_bias=True,
                dimension=dimension,
            )
        else:
            self.norm = get_norm(norm_type, up_conv_nn[1], bn_momentum=bn_momentum, D=dimension)
            self.block = get_block(block_norm_type, up_conv_nn[1], up_conv_nn[1], bn_momentum=bn_momentum, D=dimension)
            self.final = None

    def forward(self, x, x_skip):
        if x_skip is not None:
            x = ME.cat(x, x_skip)
        out_s = self.conv(x)
        if self.final is None:
            out_s = self.norm(out_s)
            out = self.block(out_s)
            return out
        else:
            out_s = MEF.relu(out_s)
            out = self.final(out_s)
            return out
