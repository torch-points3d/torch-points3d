import enum


class SchedulerUpdateOn(enum.Enum):
    ON_EPOCH = "on_epoch"
    ON_NUM_BATCH = "on_num_batch"
    ON_NUM_SAMPLE = "on_num_sample"


class ConvolutionFormat(enum.Enum):
    DENSE = "dense"
    PARTIAL_DENSE = "partial_dense"
    MESSAGE_PASSING = "message_passing"
    SPARSE = "sparse"
