import logging
import fcntl
import termios
import struct
import os

log = logging.getLogger(__name__)

def terminal_size():
    h, w, _, _ = struct.unpack('HHHH',
        fcntl.ioctl(0, termios.TIOCGWINSZ,
        struct.pack('HHHH', 0, 0, 0, 0)))
    return w, h


def breakpoint_zero():
    if os.getenv("LOCAL_RANK") == "0":
        import pdb; pdb.set_trace()

class COLORS:
    """[This class is used to color the bash shell by using {} {} {} with 'COLORS.{}, text, COLORS.END_TOKEN']
    """

    TRAIN_COLOR = "\033[0;92m"
    VAL_COLOR = "\033[0;94m"
    TEST_COLOR = "\033[0;93m"
    BEST_COLOR = "\033[0;92m"

    END_TOKEN = "\033[0m)"
    END_NO_TOKEN = "\033[0m"

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

STAGE_COLORS = {
    "train": COLORS.TRAIN_COLOR,
    "val": COLORS.VAL_COLOR,
    "test": COLORS.TEST_COLOR
}


def colored_print(color, msg):
    log.info(color + msg + COLORS.END_NO_TOKEN)

POSSIBLE_COLORS = {k:v for k,v in vars(COLORS).items() if ("\x1b" in str(v) and "\033[0m" != v)}

RANK_COLORS = {str(idx): color for idx, (color_name, color) in enumerate(POSSIBLE_COLORS.items())}

def colored_rank_print(msg):
    color = RANK_COLORS[os.getenv("LOCAL_RANK", "0")]
    log.info(color + msg + COLORS.END_NO_TOKEN)

def log_metrics(metrics, stage, depth = 0):
    is_depth_zero = depth == 0
    if is_depth_zero:
        w, _ = terminal_size()
        print("\n")
        print("=" * w)
    
    for key, value in metrics.items():
        logging_shift = " " * depth * 4
        if isinstance(value, dict):
            colored_print(STAGE_COLORS[stage], f"{logging_shift}{key.upper()}:" + "{")
            log_metrics(value, stage, depth = depth + 1)
            colored_print(STAGE_COLORS[stage], f"{logging_shift}" + "}")
        else:
            logging_shift = " " * depth * 4
            colored_print(STAGE_COLORS[stage], f"{logging_shift}{key.upper()}: {value}")
    
    if is_depth_zero:
        print("=" * w + "\n")
