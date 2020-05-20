from tqdm.auto import tqdm
from tqdm import std
import numpy as np

from torch_points3d.utils.colors import COLORS


class Coloredtqdm(tqdm):
    def set_postfix(self, ordered_dict=None, refresh=True, color=None, round=4, **kwargs):
        postfix = std._OrderedDict([] if ordered_dict is None else ordered_dict)

        for key in sorted(kwargs.keys()):
            postfix[key] = kwargs[key]

        for key in postfix.keys():
            if isinstance(postfix[key], std.Number):
                postfix[key] = self.format_num_to_k(np.round(postfix[key], round), k=round + 1)
            if isinstance(postfix[key], std._basestring):
                postfix[key] = str(postfix[key])
            if len(postfix[key]) != round:
                postfix[key] += (round - len(postfix[key])) * " "

        if color is not None:
            self.postfix = color
        else:
            self.postfix = ""

        self.postfix += ", ".join(key + "=" + postfix[key] for key in postfix.keys())
        if color is not None:
            self.postfix += COLORS.END_TOKEN

        if refresh:
            self.refresh()

    def format_num_to_k(self, seq, k=4):
        seq = str(seq)
        length = len(seq)
        out = seq + " " * (k - length) if length < k else seq
        return out if length < k else seq[:k]
