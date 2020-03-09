import enum


class ConvolutionFormat(enum.Enum):
    DENSE = "dense"
    PARTIAL_DENSE = "partial_dense"
    MESSAGE_PASSING = "message_passing"
    SPARSE = "sparse"
