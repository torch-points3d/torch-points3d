import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter


class BaseModule(nn.Module):
    """ Base module class with some basic additions to the pytorch Module class
    """

    @property
    def nb_params(self):
        """This property is used to return the number of trainable parameters for a given layer
        It is useful for debugging and reproducibility.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self._nb_params = sum([np.prod(p.size()) for p in model_parameters])
        return self._nb_params


def weight_variable(shape):
    initial = torch.empty(shape, dtype=torch.float)
    torch.nn.init.xavier_normal_(initial)
    return initial


class Identity(BaseModule):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, data):
        return data


def MLP(channels, activation=nn.LeakyReLU(0.2), bn_momentum=0.1, bias=True):
    return nn.Sequential(
        *[
            nn.Sequential(
                nn.Linear(channels[i - 1], channels[i], bias=bias),
                FastBatchNorm1d(channels[i], momentum=bn_momentum),
                activation,
            )
            for i in range(1, len(channels))
        ]
    )


class UnaryConv(BaseModule):
    def __init__(self, kernel_shape):
        """
        1x1 convolution on point cloud (we can even call it a mini pointnet)
        """
        super(UnaryConv, self).__init__()
        self.weight = Parameter(weight_variable(kernel_shape))

    def forward(self, features):
        """
        features(Torch Tensor): size N x d d is the size of inputs
        """
        return torch.matmul(features, self.weight)

    def __repr__(self):
        return "UnaryConv {}".format(self.weight.shape)


class MultiHeadClassifier(BaseModule):
    """ Allows segregated segmentation in case the category of an object is known. This is the case in ShapeNet
    for example.

        Arguments:
            in_features -- size of the input channel
            cat_to_seg {[type]} -- category to segment maps for example:
                {
                    'Airplane': [0,1,2],
                    'Table': [3,4]
                }

        Keyword Arguments:
            dropout_proba  (default: {0.5})
            bn_momentum  -- batch norm momentum (default: {0.1})
        """

    def __init__(self, in_features, cat_to_seg, dropout_proba=0.5, bn_momentum=0.1):
        super().__init__()
        self._cat_to_seg = {}
        self._num_categories = len(cat_to_seg)
        self._max_seg_count = 0
        self._max_seg = 0
        self._shifts = torch.zeros((self._num_categories,), dtype=torch.long)
        for i, seg in enumerate(cat_to_seg.values()):
            self._max_seg_count = max(self._max_seg_count, len(seg))
            self._max_seg = max(self._max_seg, max(seg))
            self._shifts[i] = min(seg)
            self._cat_to_seg[i] = seg

        self.channel_rasing = MLP(
            [in_features, self._num_categories * in_features], bn_momentum=bn_momentum, bias=False
        )
        if dropout_proba:
            self.channel_rasing.add_module("Dropout", nn.Dropout(p=dropout_proba))

        self.classifier = UnaryConv((self._num_categories, in_features, self._max_seg_count))
        self._bias = Parameter(torch.zeros(self._max_seg_count,))

    def forward(self, features, category_labels, **kwargs):
        assert features.dim() == 2
        self._shifts = self._shifts.to(features.device)
        in_dim = features.shape[-1]
        features = self.channel_rasing(features)
        features = features.reshape((-1, self._num_categories, in_dim))
        features = features.transpose(0, 1)  # [num_categories, num_points, in_dim]
        features = self.classifier(features) + self._bias  # [num_categories, num_points, max_seg]
        ind = category_labels.unsqueeze(-1).repeat(1, 1, features.shape[-1]).long()

        logits = features.gather(0, ind).squeeze(0)
        softmax = torch.nn.functional.log_softmax(logits, dim=-1)

        output = torch.zeros(logits.shape[0], self._max_seg + 1).to(features.device)
        cats_in_batch = torch.unique(category_labels)
        for cat in cats_in_batch:
            cat_mask = category_labels == cat
            seg_indices = self._cat_to_seg[cat.item()]
            probs = softmax[cat_mask, : len(seg_indices)]
            output[cat_mask, seg_indices[0] : seg_indices[-1] + 1] = probs
        return output


class FastBatchNorm1d(BaseModule):
    def __init__(self, num_features, momentum=0.1):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, momentum=momentum)

    def _forward_dense(self, x):
        return self.batch_norm(x)

    def _forward_sparse(self, x):
        """ Batch norm 1D is not optimised for 2D tensors. The first dimension is supposed to be
        the batch and therefore not very large. So we introduce a custom version that leverages BatchNorm1D
        in a more optimised way
        """
        x = x.unsqueeze(2)
        x = x.transpose(0, 2)
        x = self.batch_norm(x)
        x = x.transpose(0, 2)
        return x.squeeze()

    def forward(self, x):
        if x.dim() == 2:
            return self._forward_sparse(x)
        elif x.dim() == 3:
            return self._forward_dense(x)
        else:
            raise ValueError("Non supported number of dimensions {}".format(x.dim()))


class Seq(nn.Sequential):
    def __init__(self):
        super().__init__()
        self._num_modules = 0

    def append(self, module):
        self.add_module(str(self._num_modules), module)
        self._num_modules += 1
        return self
