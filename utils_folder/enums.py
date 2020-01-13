import enum


class ConvolutionFormat(enum.Enum):
    DENSE = (0, True, "dense")
    PARTIAL_DENSE = (1, False, "partial_dense")
    MESSAGE_PASSING = (2, False, "message_passing")
