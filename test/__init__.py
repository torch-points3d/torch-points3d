import torch


def run_if_cuda(func):
    def wrapped_func(*args, **kwargs):
        if torch.cuda.is_available():
            return func(*args, **kwargs)
        else:
            return

    return wrapped_func
