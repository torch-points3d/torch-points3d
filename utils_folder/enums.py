import enum


class CONVOLUTION_FORMAT(enum.Enum):
    DENSE = (0, True)
    PARTIAL_DENSE = (1, False)
    MESSAGE_PASSING = (2, False)
