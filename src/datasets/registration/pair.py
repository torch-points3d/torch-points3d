import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch


def make_pair(data_source: Data, data_target: Data):
    """
    add in a Data object the source elem, the target elem.
    It also add the pos, x . It is the concatenated version so the user can choose
    how to create batches.

    """
    # add concatenation of the point cloud

    batch = data_source
    for key_target in data_target.keys:
        batch[key_target+"_target"] = data_target[key_target]
    return batch.contiguous()
