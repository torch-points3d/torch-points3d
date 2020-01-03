import torch


def weight_variable(shape):

    initial = torch.empty(shape, dtype=torch.float)
    torch.nn.init.xavier_normal_(initial)
    return initial
