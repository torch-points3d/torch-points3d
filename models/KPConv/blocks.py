
# Adaption from https://github.com/humanpose1/KPConvTorch/blob/master/models/layers.py

import torch
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import max_pool
from kernels.kp_module import PointKernel
from models.layers import KPConvLayer
from models.layers import DeformableKPConvLayer
from models.layers import UnaryConv
from models.layers import max_pool


class Block(torch.nn.Module):
    """
    basic block
    method:
    - activate_feature: BN + activation
    """

    def __init__(self):
        super(Block, self).__init__()

    def forward(self, data):
        raise NotImplementedError("This is an abstract basic Block")

    def activate_feature(self, x, bn):
        """
        batch norm and activation function.
        """
        if(self.config.NETWORK.USE_BATCH_NORM):
            return self.activation(bn(x))
        else:
            return self.activation(x)


class SimpleBlock(Block):
    """
    simple layer with KPConv convolution -> activation -> BN
    we can perform a stride version (just change the query and the neighbors)
    """

    def __init__(self, num_inputs, num_outputs, layer_ind,
                 kp_conv,
                 config,
                 is_strided=False,
                 activation=torch.nn.LeakyReLU(negative_slope=0.2)):

        super(SimpleBlock, self).__init__()
        self.layer_ind = layer_ind
        self.config = config
        self.is_strided = is_strided
        self.kp_conv = kp_conv
        self.bn = torch.nn.BatchNorm1d(
            num_outputs, momentum=self.config.NETWORK.BATCH_NORM_MOMENTUM)
        self.activation = activation

    def forward(self, data):

        inputs = data.x
        if(not self.is_strided):
            x = self.kp_conv(
                pos=(data.points[self.layer_ind],
                     data.points[self.layer_ind]),
                neighbors=data.list_neigh[self.layer_ind],
                x=inputs)
        else:
            x = self.kp_conv(
                pos=(data.points[self.layer_ind],
                     data.points[self.layer_ind+1]),
                neighbors=data.list_pool[self.layer_ind],
                x=inputs)
        x = self.activate_feature(x, self.bn)
        data.x = x
        return data


class UnaryBlock(Block):

    """
    layer with  unary convolution -> activation -> BN
    """

    def __init__(self, num_inputs, num_outputs,
                 config, activation=torch.nn.LeakyReLU(negative_slope=0.2)):
        super(UnaryBlock, self).__init__()
        self.uconv = UnaryConv(num_inputs, num_outputs,
                               config)
        self.bn = torch.nn.BatchNorm1d(
            num_outputs, momentum=self.config.NETWORK.BATCH_NORM_MOMENTUM)
        self.activation = activation

    def forward(self, data):
        inputs = data.x
        x = self.uconv(inputs)
        x = self.activate_feature(x, self.bn)
        data.x = x
        return data


class ResnetBlock(Block):

    """
    layer with KPConv with residual units
    KPConv -> KPConv + shortcut
    """

    def __init__(self, num_inputs, num_outputs, radius, layer_ind,
                 config,
                 is_strided=False,
                 activation=torch.nn.LeakyReLU(negative_slope=0.2)):
        super(ResnetBlock, self).__init__()
        self.layer_ind = layer_ind
        self.is_strided = is_strided
        self.config = config

        self.size = [num_inputs, num_outputs, num_outputs, num_outputs]
        self.kp_conv0 = KPConvLayer(radius, num_inputs, num_outputs, config)
        self.kp_conv1 = KPConvLayer(radius, num_outputs, num_outputs, config)

        self.activation = activation
        self.bn0 = torch.nn.BatchNorm1d(
            num_outputs, momentum=self.config.NETWORK.BATCH_NORM_MOMENTUM)

        self.bn1 = torch.nn.BatchNorm1d(
            num_outputs, momentum=self.config.NETWORK.BATCH_NORM_MOMENTUM)

        if(num_inputs != num_outputs):
            self.shortcut_op = UnaryConv(num_inputs, num_outputs,
                                         config)
        else:
            self.shortcut_op = torch.nn.Identity()

    def forward(self, data):
        inputs = data.x
        x = self.kp_conv0(data.points[self.layer_ind],
                          data.points[self.layer_ind],
                          data.list_neigh[self.layer_ind],
                          inputs)

        x = self.activate_feature(x, self.bn0)
        if(not self.is_strided):
            x = self.kp_conv1(data.points[self.layer_ind],
                              data.points[self.layer_ind],
                              data.list_neigh[self.layer_ind],
                              x)
            x = self.activate_feature(x, self.bn1)
            data.x = x + self.shortcut_op(inputs)

        else:
            x = self.kp_conv(data.points[self.layer_ind+1],
                             data.points[self.layer_ind],
                             data.list_pool[self.layer_ind],
                             x)
            x = self.activate_feature(x, self.bn1)
            shortcut = self.shortcut_op(max_pool(
                inputs, data.list_pool[self.layer_ind]))
            data.x = x + shortcut

        return data


class ResnetBottleNeckBlock(Block):
    """
    uconv -> kpconv -> uconv + shortcut
    """

    def __init__(self, num_inputs, num_outputs, layer_ind,
                 kp_conv,
                 config,
                 is_strided=False,
                 activation=torch.nn.LeakyReLU(negative_slope=0.2)):
        super(ResnetBottleNeckBlock, self).__init__()
        self.config = config
        self.layer_ind = layer_ind
        self.is_strided = is_strided

        self.uconv0 = UnaryConv(num_inputs, num_outputs//4, config)
        # self.kp_conv = KPConvLayer(radius, self.size[1], self.size[2], config)
        self.kp_conv = kp_conv
        self.uconv1 = UnaryConv(num_outputs//4, num_outputs, config)

        self.activation = activation
        self.bn0 = torch.nn.BatchNorm1d(
            num_outputs//4, momentum=self.config.NETWORK.BATCH_NORM_MOMENTUM)

        self.bn1 = torch.nn.BatchNorm1d(
            num_outputs//4, momentum=self.config.NETWORK.BATCH_NORM_MOMENTUM)

        self.bn2 = torch.nn.BatchNorm1d(
            num_outputs, momentum=self.config.NETWORK.BATCH_NORM_MOMENTUM)

        if(num_inputs != num_outputs):
            self.shortcut_op = UnaryConv(num_inputs, num_outputs,
                                         config)
        else:
            self.shortcut_op = torch.nn.Identity()

    def forward(self, data):

        inputs = data.x
        x = self.uconv0(inputs)
        x = self.activate_feature(x, self.bn0)

        if(not self.is_strided):
            x = self.kp_conv(pos=(data.points[self.layer_ind],
                                  data.points[self.layer_ind]),
                             neighbors=data.list_neigh[self.layer_ind],
                             x=x)
            x = self.activate_feature(x, self.bn1)
        else:
            x = self.kp_conv(pos=(
                data.points[self.layer_ind],
                data.points[self.layer_ind+1]),
                neighbors=data.list_pool[self.layer_ind],
                x=x)
            x = self.activate_feature(x, self.bn1)
        x = self.uconv1(x)
        x = self.activate_feature(x, self.bn2)
        if(not self.is_strided):
            data.x = x + self.shortcut_op(inputs)
        else:
            data.x = x + self.shortcut_op(
                max_pool(inputs, data.list_pool[self.layer_ind]))
        return data


class MaxPool(torch.nn.Module):
    """
    layer that perform max_pooling
    """

    def __init__(self, layer_ind):
        super(MaxPool, self).__init__()
        self.layer_ind = layer_ind

    def forward(self, data):
        inputs = data.x
        if(data.pools[self.layer_ind].shape[1] > 2):
            x = max_pool(inputs, data.pools[self.layer_ind])
        else:
            x = None
        data.x = x
        return data


class GlobalPool(torch.nn.Module):
    """
    global pooling layer
    """

    def __init__(self):
        super(GlobalPool, self).__init__()

    def forward(self, data):

        inputs = data.x
        batch = data.list_batch[-1]
        if(len(inputs) != len(batch)):
            raise Exception("Error, the batch and the features have not the same size")

        x = global_mean_pool(inputs, batch)
        data.x = x
        return data


class MLPClassifier(Block):
    """
    two layer of MLP multi class classification
    """

    def __init__(self, num_inputs, num_classes, config,
                 num_hidden=1024,
                 dropout_prob=0.5,
                 activation=torch.nn.LeakyReLU(negative_slope=0.2)):
        super(MLPClassifier, self).__init__()
        self.config = config
        self.lin0 = torch.nn.Linear(num_inputs, num_hidden)
        self.bn0 = torch.nn.BatchNorm1d(
            num_hidden,
            momentum=self.config.NETWORK.BATCH_NORM_MOMENTUM)

        self.dropout = torch.nn.Dropout(p=dropout_prob)

        self.lin1 = torch.nn.Linear(num_hidden, num_classes)
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.activation = activation

    def forward(self, data):

        inputs = data.x

        x = self.activate_feature(self.lin0(inputs), self.bn0)
        x = self.dropout(x)
        x = self.lin1(x)
        x = self.softmax(x)
        data.x = x
        return data
