import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicLinear(nn.Module):
    def __init__(self, inc, outc, bias=True):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.bias = bias
        self.linear = nn.Linear(inc, outc, bias=bias)
        self.runtime_inc = None
        self.runtime_outc = None
        self.runtime_inc_constraint = None

    def __repr__(self):
        return 'DynamicLinear(inc=%d, outc=%d)' % (self.inc, self.outc)

    def set_in_channel(self, in_channel=None, constraint=None):
        if in_channel is not None:
            self.runtime_inc = in_channel
        elif constraint is not None:
            self.runtime_inc_constraint = torch.from_numpy(
                np.array(constraint)).long()
        else:
            raise NotImplementedError

    def set_output_channel(self, out_channel):
        self.runtime_outc = out_channel

    def forward(self, inputs):
        assert self.runtime_outc is not None
        c = inputs.shape[-1]
        big_weight = self.linear.weight
        if self.runtime_inc_constraint is None:
            weight = big_weight[:, :c]
        else:
            weight = big_weight[:, self.runtime_inc_constraint]
        weight = weight[:self.runtime_outc, :].transpose(0, 1).contiguous()
        if not self.bias:
            return torch.mm(inputs, weight)
        else:
            return torch.mm(inputs,
                            weight) + self.linear.bias[:self.runtime_outc]


class DynamicBatchNorm(nn.Module):
    SET_RUNNING_STATISTICS = False

    def __init__(self, c, cr_bounds=[0.25, 1.0], eps=1e-5, momentum=0.1):
        super().__init__()
        self.c = c
        self.eps = eps
        self.momentum = momentum
        self.cr_bounds = cr_bounds
        self.bn = nn.BatchNorm1d(c, eps=eps, momentum=momentum)
        self.channels = []
        self.runtime_channel = None

    def __repr__(self):
        return 'DynamicBatchNorm(cmax=%d)' % (self.c)

    def set_channel(self, channel):
        self.runtime_channel = channel

    def bn_forward(self, x, bn, feature_dim):
        if bn.num_features == feature_dim or DynamicBatchNorm.SET_RUNNING_STATISTICS:
            return bn(x)
        else:
            exponential_average_factor = 0.0

            if bn.training and bn.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(
                            bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum
            return F.batch_norm(
                x,
                bn.running_mean[:feature_dim],
                bn.running_var[:feature_dim],
                bn.weight[:feature_dim],
                bn.bias[:feature_dim],
                bn.training or not bn.track_running_stats,
                exponential_average_factor,
                bn.eps,
            )

    def forward(self, inputs):
        return self.bn_forward(inputs, self.bn, inputs.shape[-1])
