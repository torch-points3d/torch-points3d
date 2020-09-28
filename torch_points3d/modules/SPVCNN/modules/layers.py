import copy
import random
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn

import torchsparse.nn as spnn

from torch_points3d.modules.SPVCNN.modules.dynamic_sparseop import *
from torch_points3d.modules.SPVCNN.modules.dynamic_op import *
from torch_points3d.modules.SPVCNN.modules.modules import RandomDepth, RandomModule


def adjust_bn_according_to_idx(bn, idx):
    bn.weight.data = torch.index_select(bn.weight.data, 0, idx)
    bn.bias.data = torch.index_select(bn.bias.data, 0, idx)
    bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, idx)
    bn.running_var.data = torch.index_select(bn.running_var.data, 0, idx)


class LinearBlock(nn.Module):
    def __init__(self, inc, outc, bias=True, no_relu=False, no_bn=False):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.no_relu = no_relu
        self.bias = bias
        self.no_bn = no_bn
        net = OrderedDict([('conv', nn.Linear(inc, outc, bias=bias))])
        if not self.no_bn:
            net['bn'] = nn.BatchNorm1d(outc)
        if not self.no_relu:
            net['act'] = nn.ReLU(True)

        self.net = nn.Sequential(net)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_weights(self, nas_module, runtime_inc_constraint=None):
        cur_kernel = nas_module.net.conv.linear.weight
        if runtime_inc_constraint is None:
            cur_kernel = cur_kernel[:, :self.inc]
        else:
            cur_kernel = cur_kernel[:, runtime_inc_constraint]
        cur_kernel = cur_kernel[:self.outc, :]
        self.net.conv.weight.data = cur_kernel

        if self.bias:
            cur_bias = nas_module.net.conv.linear.bias
            cur_bias = cur_bias[:self.outc]
            self.net.conv.bias.data = cur_bias

        if not self.no_bn:
            self.net.bn.weight.data = nas_module.net.bn.bn.weight[:self.outc]
            self.net.bn.running_var.data = nas_module.net.bn.bn.running_var[:
                                                                            self
                                                                            .
                                                                            outc]
            self.net.bn.running_mean.data = nas_module.net.bn.bn.running_mean[:
                                                                              self
                                                                              .
                                                                              outc]
            self.net.bn.bias.data = nas_module.net.bn.bn.bias[:self.outc]
            self.net.bn.num_batches_tracked.data = nas_module.net.bn.bn.num_batches_tracked

    def forward(self, inputs):
        return self.net(inputs)


class DynamicLinearBlock(RandomModule):
    def __init__(self,
                 inc,
                 outc,
                 cr_bounds=[0.25, 1.0],
                 bias=True,
                 no_relu=False,
                 no_bn=False):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.bias = bias
        self.cr_bounds = cr_bounds
        self.no_relu = no_relu
        self.no_bn = no_bn

        net = OrderedDict([('conv', DynamicLinear(inc, outc, bias=bias))])
        if not self.no_bn:
            net['bn'] = DynamicBatchNorm(outc)
        if not self.no_relu:
            net['act'] = nn.ReLU(True)

        self.net = nn.Sequential(net)
        self.runtime_inc = None
        self.runtime_outc = None
        self.in_channel_constraint = None

    def re_organize_middle_weights(self):
        weights = self.net.conv.linear.weight.data
        outc, inc = weights.shape
        importance = torch.sum(torch.abs(weights), dim=(1))

        sorted_importance, sorted_idx = torch.sort(importance,
                                                   dim=0,
                                                   descending=True)
        self.net.conv.linear.weight.data = torch.index_select(
            self.net.conv.linear.weight.data, 0, sorted_idx)
        if self.bias:
            self.net.conv.linear.bias.data = torch.index_select(
                self.net.conv.linear.bias.data, 0, sorted_idx)
        adjust_bn_according_to_idx(self.net.bn.bn, sorted_idx)

    def constrain_in_channel(self, constraint):
        self.in_channel_constraint = constraint
        self.runtime_inc = None

    def manual_select(self, channel):
        self.net.conv.set_output_channel(channel)
        if not self.no_bn:
            self.net.bn.set_channel(channel)
        self.runtime_outc = channel

    def manual_select_in(self, channel):
        self.runtime_inc = channel

    def random_sample(self):
        cr = random.uniform(*self.cr_bounds)
        channel = make_divisible(int(cr * self.outc))
        self.net.conv.set_output_channel(channel)
        if not self.no_bn:
            self.net.bn.set_channel(channel)
        self.runtime_outc = channel
        return channel

    def clear_sample(self):
        self.runtime_outc = None

    def status(self):
        return self.runtime_outc

    def determinize(self):
        assert self.runtime_inc is not None or self.in_channel_constraint is not None

        inc = self.runtime_inc if self.runtime_inc is not None \
            else len(self.in_channel_constraint)

        determinized_model = LinearBlock(inc,
                                         self.runtime_outc,
                                         bias=self.bias,
                                         no_relu=self.no_relu,
                                         no_bn=self.no_bn)
        determinized_model.load_weights(self, self.in_channel_constraint)
        return determinized_model

    def forward(self, x):
        if self.in_channel_constraint is None:
            in_channel = x.shape[-1]
            self.runtime_inc = in_channel
            self.net.conv.set_in_channel(in_channel=in_channel)
        else:
            self.net.conv.set_in_channel(constraint=self.in_channel_constraint)
        out = self.net(x)
        return out


class ConvolutionBlock(nn.Module):
    def __init__(self,
                 inc,
                 outc,
                 ks=3,
                 stride=1,
                 dilation=1,
                 no_relu=False,
                 transpose=False):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.ks = ks
        self.no_relu = no_relu
        self.net = nn.Sequential(
            OrderedDict([
                ('conv',
                 spnn.Conv3d(inc,
                                      outc,
                                      kernel_size=ks,
                                      dilation=dilation,
                                      stride=stride,
                                      transpose=transpose)),
                ('bn', spnn.BatchNorm(outc)),
                ('act',
                 spnn.ReLU(True) if not self.no_relu else nn.Sequential())
            ]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_weights(self, nas_module, runtime_inc_constraint=None):
        cur_kernel = nas_module.net.conv.kernel
        if runtime_inc_constraint is not None:
            cur_kernel = cur_kernel[:,
                                    runtime_inc_constraint, :] if self.ks > 1 else cur_kernel[
                                        runtime_inc_constraint]
        else:
            cur_kernel = cur_kernel[:, torch.arange(
                self.inc), :] if self.ks > 1 else cur_kernel[torch.arange(
                    self.inc)]

        cur_kernel = cur_kernel[..., torch.arange(self.outc)]
        self.net.conv.kernel.data = cur_kernel
        self.net.bn.weight.data = nas_module.net.bn.bn.weight[:self.outc]
        self.net.bn.running_var.data = nas_module.net.bn.bn.running_var[:
                                                                           self
                                                                           .
                                                                           outc]
        self.net.bn.running_mean.data = nas_module.net.bn.bn.running_mean[:
                                                                             self
                                                                             .
                                                                             outc]
        self.net.bn.bias.data = nas_module.net.bn.bn.bias[:self.outc]
        self.net.bn.num_batches_tracked.data = nas_module.net.bn.bn.num_batches_tracked

    def forward(self, inputs):
        return self.net(inputs)


class DynamicConvolutionBlock(RandomModule):
    def __init__(self,
                 inc,
                 outc,
                 cr_bounds=[0.25, 1.0],
                 ks=3,
                 stride=1,
                 dilation=1,
                 no_relu=False):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.ks = ks
        self.s = stride
        self.cr_bounds = cr_bounds
        self.no_relu = no_relu
        self.net = nn.Sequential(
            OrderedDict([
                ('conv',
                 SparseDynamicConv3d(inc,
                                             outc,
                                             kernel_size=ks,
                                             dilation=dilation,
                                             stride=stride)),
                ('bn', SparseDynamicBatchNorm(outc)),
                ('act',
                 spnn.ReLU(True) if not self.no_relu else nn.Sequential())
            ]))
        self.runtime_inc = None
        self.runtime_outc = None
        self.in_channel_constraint = None

    def re_organize_middle_weights(self):
        weights = self.net.conv.kernel.data
        if len(weights.shape) == 3:
            k, inc, outc = weights.shape
            importance = torch.sum(torch.abs(weights), dim=(0, 1))
        else:
            inc, outc = weights.shape
            importance = torch.sum(torch.abs(weights), dim=(0))

        sorted_importance, sorted_idx = torch.sort(importance,
                                                   dim=0,
                                                   descending=True)
        if len(weights.shape) == 3:
            self.net.conv.kernel.data = torch.index_select(
                self.net.conv.kernel.data, 2, sorted_idx)
        else:
            self.net.conv.kernel.data = torch.index_select(
                self.net.conv.kernel.data, 1, sorted_idx)
        adjust_bn_according_to_idx(self.net.bn.bn, sorted_idx)

    def constrain_in_channel(self, constraint):
        self.in_channel_constraint = constraint
        self.runtime_inc = None

    def manual_select(self, channel):
        self.net.conv.set_output_channel(channel)
        self.net.bn.set_channel(channel)
        self.runtime_outc = channel

    def manual_select_in(self, channel):
        if self.in_channel_constraint is not None:
            return
        self.runtime_inc = channel

    def random_sample(self):
        cr = random.uniform(*self.cr_bounds)
        channel = make_divisible(int(cr * self.outc))
        self.net.conv.set_output_channel(channel)
        self.net.bn.set_channel(channel)
        self.runtime_outc = channel
        return channel

    def clear_sample(self):
        self.runtime_outc = None

    def status(self):
        return self.runtime_outc

    def determinize(self):
        if self.runtime_inc is None:
            assert self.in_channel_constraint is not None
            inc = len(self.in_channel_constraint)
        else:
            inc = self.runtime_inc

        determinized_model = ConvolutionBlock(inc,
                                              self.runtime_outc,
                                              self.ks,
                                              self.s,
                                              no_relu=self.no_relu)
        determinized_model.load_weights(self, self.in_channel_constraint)
        return determinized_model

    def forward(self, x):
        if self.in_channel_constraint is None:
            in_channel = x.F.shape[-1]
            self.runtime_inc = in_channel
            self.net.conv.set_in_channel(in_channel=in_channel)
        else:
            self.net.conv.set_in_channel(constraint=self.in_channel_constraint)

        out = self.net(x)
        return out


class DynamicDeconvolutionBlock(RandomModule):
    def __init__(self, inc, outc, cr_bounds=[0.25, 1.0], ks=3, stride=1):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.ks = ks
        self.s = stride
        self.cr_bounds = cr_bounds
        self.net = nn.Sequential(
            OrderedDict([('conv',
                          SparseDynamicConv3d(inc,
                                                      outc,
                                                      kernel_size=ks,
                                                      stride=stride,
                                                      transpose=True)),
                         ('bn', SparseDynamicBatchNorm(outc)),
                         ('act', spnn.ReLU(True))]))
        self.runtime_inc = None
        self.runtime_outc = None
        self.in_channel_constraint = None

    def manual_select(self, channel):
        self.net.conv.set_output_channel(channel)
        self.net.bn.set_channel(channel)
        self.runtime_outc = channel

    def manual_select_in(self, channel):
        if self.in_channel_constraint is not None:
            return
        self.runtime_inc = channel

    def random_sample(self):
        cr = random.uniform(*self.cr_bounds)
        channel = make_divisible(int(cr * self.outc))
        self.net.conv.set_output_channel(channel)
        self.net.bn.set_channel(channel)
        self.runtime_outc = channel
        return channel

    def clear_sample(self):
        self.runtime_outc = None

    def status(self):
        return self.runtime_outc

    def determinize(self):
        determinized_model = ConvolutionBlock(self.runtime_inc,
                                              self.runtime_outc,
                                              self.ks,
                                              self.s,
                                              transpose=True)
        determinized_model.load_weights(self, self.in_channel_constraint)
        return determinized_model

    def forward(self, x):
        in_channel = x.F.shape[-1]
        self.runtime_inc = in_channel
        self.net.conv.set_in_channel(in_channel=in_channel)

        out = self.net(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, net, downsample):
        self.net = net
        self.downsample = downsample
        self.relu = spnn.ReLU(True)

    def forward(self, inputs):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class DynamicResidualBlock(nn.Module):
    def __init__(self,
                 inc,
                 outc,
                 cr_bounds=[0.25, 1.0],
                 ks=3,
                 stride=1,
                 dilation=1):
        # make sure first run random_sample, then constrain_output_channel
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.cr_bounds = cr_bounds
        self.stride = stride

        self.use_skip_conn = (inc == outc and stride == 1)
        self.net_depth = None

        # can separate the last layer from self.net
        self.net = RandomDepth(*[
            DynamicConvolutionBlock(inc, outc, cr_bounds, ks, stride, dilation,
                                    False),
            DynamicConvolutionBlock(outc, outc, cr_bounds, ks, stride,
                                    dilation, True)
        ],
                               depth_min=2)

        self.downsample = nn.Sequential() if self.use_skip_conn else \
                DynamicConvolutionBlock(inc, outc, cr_bounds, ks=1, stride=1, dilation=1, no_relu=True)

        self.relu = spnn.ReLU(True)
        self.runtime_inc = None

    def constrain_output_channel(self, output_channel):
        assert self.net_depth is not None or self.net.depth is not None
        if self.net_depth is None:
            self.net_depth = self.net.depth
        if not self.use_skip_conn:
            self.downsample.manual_select(output_channel)

        self.net.layers[self.net_depth - 1].manual_select(output_channel)

    def clear_sample(self):
        for name, module in self.named_modules():
            if isinstance(module, RandomModule):
                module.clear_sample()

    def random_sample(self):
        self.net_depth = self.net.random_sample()
        for i in range(self.net_depth - 1):
            self.net.layers[i].random_sample()

        for i in range(1, self.net_depth):
            self.net.layers[i].manual_select_in(self.net.layers[i -
                                                                1].status())

    def manual_select_in(self, channel):
        self.runtime_inc = channel
        self.net.layers[0].manual_select_in(channel)
        if self.use_skip_conn:
            self.downsample.manual_select_in(channel)

    def manual_select(self, sample):
        for name, module in self.named_random_modules():
            module.manual_select(sample[name])
        self.net_depth = self.net.depth
        for i in range(1, self.net_depth):
            self.net.layers[i].manual_select_in(self.net.layers[i -
                                                                1].status())

    def determinize(self):
        net = []
        for i in range(self.net.depth):
            net.append(self.net.layers[i].determinize())
            net[-1].load_weights(self.net.layers[i])

        net = nn.Sequential(*net)
        downsample = nn.Sequential(
        ) if not self.use_skip_conn else self.downsample.determinize()
        return ResidualBlock(net, downsample)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out
