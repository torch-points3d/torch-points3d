from typing import Any
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch_points3d.datasets.segmentation import IGNORE_LABEL
from .metric_losses import *


def filter_valid(output, target, ignore_label=IGNORE_LABEL, other=None):
    """ Removes predictions for nodes without ground truth """
    idx = target != ignore_label
    if other is not None:
        return output[idx, :], target[idx], other[idx, ...]
    return output[idx, :], target[idx]


class LossAnnealer(torch.nn.modules.loss._Loss):
    """
    This class will be used to perform annealing between two losses
    """

    def __init__(self, args):
        super(LossAnnealer, self).__init__()
        self._coeff = 0.5  # args.coefficient
        self.normalized_loss = True

    def forward(self, loss_1, loss_2, **kwargs):
        annealing_alpha = kwargs.get("annealing_alpha", None)
        if annealing_alpha is None:
            return self._coeff * loss_1 + (1 - self._coeff) * loss_2
        else:
            return (1 - annealing_alpha) * loss_1 + annealing_alpha * loss_2


class LossFactory(torch.nn.modules.loss._Loss):
    def __init__(self, loss, dbinfo):
        super(LossFactory, self).__init__()

        self._loss = loss
        self.special_args = {}
        self.search_for_args = []
        if self._loss == "cross_entropy":
            self._loss_func = nn.functional.cross_entropy
            self.special_args = {"weight": dbinfo["class_weights"]}
            # self.search_for_args = ['cloug_flag']

        elif self._loss == "focal_loss":
            self._loss_func = FocalLoss(alphas=dbinfo["class_weights"])

        elif self._loss == "KLDivLoss":
            self._loss_func = WrapperKLDivLoss()
            self.search_for_args = ["segm_size", "label_vec"]

        else:
            raise NotImplementedError

    def forward(self, input, target, **kwargs):
        added_arguments = OrderedDict()
        for key in self.search_for_args:
            added_arguments[key] = kwargs.get(key, None)
        input, target = filter_valid(input, target)
        return self._loss_func(input, target, **added_arguments, **self.special_args)


class FocalLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self, gamma: float = 2, alphas: Any = None, size_average: bool = True, normalized: bool = True,
    ):
        super(FocalLoss, self).__init__()
        self._gamma = gamma
        self._alphas = alphas
        self.size_average = size_average
        self.normalized = normalized

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = torch.gather(logpt, -1, target.unsqueeze(-1))
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self._alphas is not None:
            at = self._alphas.gather(0, target)
            logpt = logpt * Variable(at)

        if self.normalized:
            sum_ = 1 / torch.sum((1 - pt) ** self._gamma)
        else:
            sum_ = 1

        loss = -1 * sum_ * (1 - pt) ** self._gamma * logpt
        return loss.sum()


class WrapperKLDivLoss(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        super(WrapperKLDivLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, label_vec=None, segm_size=None):
        label_vec = Variable(label_vec).float() / segm_size.unsqueeze(-1).float()
        input = F.log_softmax(input, dim=-1)
        loss = torch.nn.modules.loss.KLDivLoss()(input, label_vec)
        return loss
