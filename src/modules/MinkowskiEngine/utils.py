import torch
import MinkowskiEngine as ME
import torch.nn.functional as F
import re
from torch_scatter import scatter_add, scatter_mean


def shuffle_data(data):
    num_points = data.pos.shape[0]
    shuffle_idx = torch.randperm(num_points)
    for key in set(data.keys):
        item = data[key]
        if num_points == item.shape[0]:
            data[key] = item[shuffle_idx]
    return data


def quantize_data(data, mode="last"):
    """ Creates the quantized version of a data object in which ``pos`` has
    already been quantized. It either averages all points in the same cell or picks
    the last one as being the representent for that cell.

    Parameters
    ----------
    data : Data
        data object in which ``pos`` has already been quantized. ``pos`` must be
        a ``torch.Tensor[int]`` or ``torch.Tensor[long]``
    mode : str
        Option to select how the features and labels for each voxel is computed. Can be ``last`` or ``mean``.
        ``last`` selects the last point falling in a voxel as the representent, ``mean`` takes the average.

    Returns
    -------
    data: Data
        Returns the same data object with only one point per voxel
    """
    assert data.pos.dtype == torch.int or data.pos.dtype == torch.long

    # Build clusters
    if hasattr(data, "batch"):
        pos = torch.cat([data.batch.unsqueeze(-1), data.pos], dim=-1)
    else:
        pos = data.pos.clone()
    unique_pos, cluster = torch.unique(pos, return_inverse=True, dim=0)
    unique_pos_indices = torch.arange(cluster.size(0), dtype=cluster.dtype, device=cluster.device)
    unique_pos_indices = cluster.new_empty(unique_pos.size(0)).scatter_(0, cluster, unique_pos_indices)

    # Agregate features within the same voxel
    data = groud_data(data, cluster, unique_pos_indices, mode=mode)
    return data


def groud_data(data, cluster, unique_pos_indices, mode="last"):
    num_nodes = data.num_nodes
    for key, item in data:
        if bool(re.search("edge", key)):
            raise ValueError("Edges not supported. Wrong data type.")

        if torch.is_tensor(item) and item.size(0) == num_nodes:
            if mode == "last" or key == "batch":
                data[key] = item[unique_pos_indices]
            elif mode == "mean":
                if key == "y":
                    item = F.one_hot(item)
                    item = scatter_add(item, cluster, dim=0)
                    data[key] = item.argmax(dim=-1)
                else:
                    data[key] = scatter_mean(item, cluster, dim=0)
    return data


def to_sparse_input(
    data, grid_size, save_delta=True, save_delta_norm=True, mode="last", quantizing_func=torch.floor,
):
    if quantizing_func not in [torch.floor, torch.ceil, torch.round]:
        raise Exception("quantizing_func should be floor, ceil, round")

    if grid_size is None or grid_size <= 0:
        raise Exception("Grid size should be provided and greater than 0")

    # Quantize positions
    raw_pos = data.pos.clone()
    data.pos = quantizing_func(data.pos / grid_size).int()

    # Add delta as a feature
    if save_delta or save_delta_norm:
        normalised_delta = compute_sparse_delta(raw_pos, data.pos, grid_size, quantizing_func)
        if save_delta:
            data.delta = normalised_delta
        if save_delta_norm:
            data.delta_norm = torch.norm(normalised_delta, p=2, dim=-1)  # normalise between -1 and 1 (roughly)

    # Agregate
    data = quantize_data(data, mode=mode)
    return data


def compute_sparse_delta(raw_pos, quantized_pos, grid_size, quantizing_func):
    """ Computes the error between the raw position and the quantized position
    Error is normalised between -1 and 1

    Parameters
    ----------
    raw_pos : torch.Tensor
    quantized_pos : torch.Tensor
    quantizing_func : func


    Returns
    -------
    torch.Tensor
        Error normalized between -1 and 1
    """
    delta = raw_pos - quantized_pos
    shift = 0
    if quantizing_func == torch.ceil:
        shift = 1
    elif quantizing_func == torch.floor:
        shift = -1

    return 2 * delta / grid_size + shift  # normalise between -1 and 1 (roughly)
