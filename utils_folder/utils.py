import numpy as np
from typing import List
import matplotlib.pyplot as plt
import os
from os import path as osp
import torch
from collections import namedtuple
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
import logging

from utils_folder.enums import ConvolutionFormat

log = logging.getLogger(__name__)


def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ["preds", "loss"])

    def model_fn(model, data, epoch=0, eval=False):
        with torch.set_grad_enabled(not eval):
            x, pos, labels = data
            x = x.to("cuda", non_blocking=True)
            pos = pos.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)

            preds = model(x, pos)
            loss = criterion(preds.view(labels.numel(), -1), labels.view(-1))
            loss.backward()
            model.optim_step()

        return ModelReturn(preds, loss.item())

    return model_fn


def set_format(model_config, cfg_training):
    """ Adds the type of convolution (DENSE, PARTIAL_DENSE, MESSAGE_PASSING)
    to the training configuration
    """
    conv_type = getattr(model_config, "conv_type", None)
    if conv_type not in [d.name for d in ConvolutionFormat]:
        raise Exception("The format type should be defined within {}".format([d.name for d in ConvolutionFormat]))
    else:
        format_conf = OmegaConf.create(
            {"conv_type": conv_type.lower(), "use_torch_loader": ConvolutionFormat[conv_type].value[1]}
        )
        return OmegaConf.merge(cfg_training, format_conf)


def merges_in_sub(x, list_conf: List):
    dict_ = {}
    for o, v in x.items():
        name = str(o)
        if isinstance(v, DictConfig):
            for c in list_conf:
                v = OmegaConf.merge(v, c)
            dict_[name] = v
        else:
            dict_[name] = v
    return OmegaConf.create(dict_)


def colored_print(color, msg):
    log.info(color + msg + "\033[0m")


def save_confusion_matrix(cm, path2save, ordered_names):
    import seaborn as sns

    sns.set(font_scale=5)

    template_path = os.path.join(path2save, "{}.svg")
    # PRECISION
    cmn = cm.astype("float") / cm.sum(axis=-1)[:, np.newaxis]
    cmn[np.isnan(cmn) | np.isinf(cmn)] = 0
    fig, ax = plt.subplots(figsize=(31, 31))
    g = sns.heatmap(
        cmn, annot=True, fmt=".2f", xticklabels=ordered_names, yticklabels=ordered_names, annot_kws={"size": 20}
    )
    # g.set_xticklabels(g.get_xticklabels(), rotation = 35, fontsize = 20)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    path_precision = template_path.format("precision")
    plt.savefig(path_precision, format="svg")

    # RECALL
    cmn = cm.astype("float") / cm.sum(axis=0)[np.newaxis, :]
    cmn[np.isnan(cmn) | np.isinf(cmn)] = 0
    fig, ax = plt.subplots(figsize=(31, 31))
    g = sns.heatmap(
        cmn, annot=True, fmt=".2f", xticklabels=ordered_names, yticklabels=ordered_names, annot_kws={"size": 20}
    )
    # g.set_xticklabels(g.get_xticklabels(), rotation = 35, fontsize = 20)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    path_recall = template_path.format("recall")
    plt.savefig(path_recall, format="svg")
