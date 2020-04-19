import torch.nn

from torch_points3d.utils.config import is_dict


def get_activation(act_opt, create_cls=True):
    if is_dict(act_opt):
        act_opt = dict(act_opt)
        act = getattr(torch.nn, act_opt["name"])
        del act_opt["name"]
        args = dict(act_opt)
    else:
        act = getattr(torch.nn, act_opt)
        args = {}

    if create_cls:
        return act(**args)
    else:
        return act
