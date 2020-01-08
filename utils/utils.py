import numpy as np
import matplotlib.pyplot as plt
import os
from os import path as osp
import torch
from collections import namedtuple


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


def set_format(model_config, cfg, available_format):
    format_type = getattr(model_config, "format_type", None)
    if format_type not in available_format.keys():
        raise Exception("The format type should be defined within {}".format(available_format.keys()))
    else:
        cfg.training.format_type = format_type
        cfg.training.use_torch_loader = available_format[format_type]


def get_log_dir(log_dir, experiment_name):
    hydra_generated_dir = os.getcwd()
    splits = hydra_generated_dir.split('/')
    if log_dir == "":
        parent_log_dir = '/'.join(splits[:-1])
    else:
        parent_log_dir = log_dir
    return osp.join(parent_log_dir, experiment_name if experiment_name != "" else splits[-1])


def colored_print(color, msg):
    print(color + msg + '\033[0m')


def save_confusion_matrix(cm, path2save, ordered_names):
    import seaborn as sns
    sns.set(font_scale=5)

    template_path = os.path.join(path2save, "{}.svg")
    # PRECISION
    cmn = cm.astype('float') / cm.sum(axis=-1)[:, np.newaxis]
    cmn[np.isnan(cmn) | np.isinf(cmn)] = 0
    fig, ax = plt.subplots(figsize=(31, 31))
    g = sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=ordered_names,
                    yticklabels=ordered_names, annot_kws={"size": 20})
    #g.set_xticklabels(g.get_xticklabels(), rotation = 35, fontsize = 20)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    path_precision = template_path.format("precision")
    plt.savefig(path_precision, format="svg")

    # RECALL
    cmn = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    cmn[np.isnan(cmn) | np.isinf(cmn)] = 0
    fig, ax = plt.subplots(figsize=(31, 31))
    g = sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=ordered_names,
                    yticklabels=ordered_names, annot_kws={"size": 20})
    #g.set_xticklabels(g.get_xticklabels(), rotation = 35, fontsize = 20)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    path_recall = template_path.format("recall")
    plt.savefig(path_recall, format="svg")
