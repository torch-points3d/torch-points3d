from tqdm import tqdm, std
from collections import OrderedDict
import numpy as np


class COLORS:
    TRAIN_COLOR = "\033[0;92m"
    VAL_COLOR = "\033[0;94m"
    TEST_COLOR = "\033[0;93m"
    BEST_COLOR = "\033[0;92m"

    END_TOKEN = "\033[0m)"

    Black = "\033[0;30m"  # Black
    Red = "\033[0;31m"  # Red
    Green = "\033[0;32m"  # Green
    Yellow = "\033[0;33m"  # Yellow
    Blue = "\033[0;34m"  # Blue
    Purple = "\033[0;35m"  # Purple
    Cyan = "\033[0;36m"  # Cyan
    White = "\033[0;37m"  # White

    # Bold
    BBlack = "\033[1;30m"  # Black
    BRed = "\033[1;31m"  # Red
    BGreen = "\033[1;32m"  # Green
    BYellow = "\033[1;33m"  # Yellow
    BBlue = "\033[1;34m"  # Blue
    BPurple = "\033[1;35m"  # Purple
    BCyan = "\033[1;36m"  # Cyan
    BWhite = "\033[1;37m"  # White

    # Underline
    UBlack = "\033[4;30m"  # Black
    URed = "\033[4;31m"  # Red
    UGreen = "\033[4;32m"  # Green
    UYellow = "\033[4;33m"  # Yellow
    UBlue = "\033[4;34m"  # Blue
    UPurple = "\033[4;35m"  # Purple
    UCyan = "\033[4;36m"  # Cyan
    UWhite = "\033[4;37m"  # White

    # Background
    On_Black = "\033[40m"  # Black
    On_Red = "\033[41m"  # Red
    On_Green = "\033[42m"  # Green
    On_Yellow = "\033[43m"  # Yellow
    On_Blue = "\033[44m"  # Blue
    On_Purple = "\033[45m"  # Purple
    On_Cyan = "\033[46m"  # Cyan
    On_White = "\033[47m"  # White

    # High Intensty
    IBlack = "\033[0;90m"  # Black
    IRed = "\033[0;91m"  # Red
    IGreen = "\033[0;92m"  # Green
    IYellow = "\033[0;93m"  # Yellow
    IBlue = "\033[0;94m"  # Blue
    IPurple = "\033[0;95m"  # Purple
    ICyan = "\033[0;96m"  # Cyan
    IWhite = "\033[0;97m"  # White

    # Bold High Intensty
    BIBlack = "\033[1;90m"  # Black
    BIRed = "\033[1;91m"  # Red
    BIGreen = "\033[1;92m"  # Green
    BIYellow = "\033[1;93m"  # Yellow
    BIBlue = "\033[1;94m"  # Blue
    BIPurple = "\033[1;95m"  # Purple
    BICyan = "\033[1;96m"  # Cyan
    BIWhite = "\033[1;97m"  # White

    # High Intensty backgrounds
    On_IBlack = "\033[0;100m"  # Black
    On_IRed = "\033[0;101m"  # Red
    On_IGreen = "\033[0;102m"  # Green
    On_IYellow = "\033[0;103m"  # Yellow
    On_IBlue = "\033[0;104m"  # Blue
    On_IPurple = "\033[10;95m"  # Purple
    On_ICyan = "\033[0;106m"  # Cyan
    On_IWhite = "\033[0;107m"  # White


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
