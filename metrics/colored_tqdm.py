from tqdm import tqdm, std
from collections import OrderedDict
import numpy as np


class COLORS:
    TRAIN_COLOR = "\033[0;92m"
    VAL_COLOR = "\033[0;94m"
    TEST_COLOR = "\033[0;93m"
    BEST_COLOR = "\033[0;92m"


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
            self.postfix += "\033[0m"

        if refresh:
            self.refresh()

    def format_num_to_k(self, seq, k=4):
        seq = str(seq)
        length = len(seq)
        out = seq + " " * (k - length) if length < k else seq
        return out if length < k else seq[:k]
