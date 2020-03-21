import numpy as np
import torch
import MinkowskiEngine as ME
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.pool.consecutive import consecutive_cluster
import re
from torch_scatter import scatter_add, scatter_mean


def shuffle_data(data):
    num_points = data.pos.shape[0]
    shuffle_idx = torch.randperm(num_points)
    new_data = Data()
    for key in set(data.keys):
        item = data[key].clone()
        if num_points == item.shape[0]:
            setattr(new_data, key, item[shuffle_idx])
    return data


def remove_duplicates_func(data):
    indices = data.indices
    num_points = indices.shape[0]
    _, inds = np.unique(indices.numpy(), axis=0, return_index=True)
    inds = torch.from_numpy(inds)
    new_data = Data()
    for key in set(data.keys):
        item = data[key].clone()
        if num_points == item.shape[0]:
            setattr(new_data, key, item[inds])
    return new_data


def apply_mean_func(data):
    num_nodes = data.num_nodes
    indices = data.indices
    _, inds = np.unique(indices.numpy(), axis=0, return_index=True)
    indices -= indices.min(0)[0]
    indices = indices.numpy()
    spatial_size = indices.max(0)
    cluster = ((indices) * np.cumsum(spatial_size)).sum(-1)
    cluster, perm = consecutive_cluster(torch.from_numpy(cluster))

    for key, item in data:
        if bool(re.search("edge", key)):
            raise ValueError("GridSampling does not support coarsening of edges")

        if torch.is_tensor(item) and item.size(0) == num_nodes:
            if key == "y":
                item = F.one_hot(item)
                item = scatter_add(item, cluster, dim=0)
                data[key] = item.argmax(dim=-1)
            elif key == "batch":
                data[key] = item[perm]
            else:
                data[key] = scatter_mean(item, cluster, dim=0)
    return data


def to_sparse_input(
    data,
    grid_size,
    save_delta=True,
    save_delta_norm=True,
    remove_duplicates=True,
    apply_mean=True,
    mapping_func=np.floor,
):

    if mapping_func not in [np.floor, np.ceil, np.rint]:
        raise Exception("mapping_func should be floor, ceil, rint")

    if apply_mean and remove_duplicates:
        raise Exception("remove_duplicates and apply_mean can't at the same time")

    if grid_size is None:
        raise Exception("Grid size should be provided")

    elif grid_size == 0:
        raise Exception("Grid size should not be equal to 0")

    else:
        num_points = data.pos.shape[0]
        quantized_coords = mapping_func(data.pos / grid_size)

        if remove_duplicates:
            inds = ME.utils.sparse_quantize(quantized_coords, return_index=True)
            indices = quantized_coords[inds]
            new_data = Data(indices=indices.long())
        else:
            new_data = Data(indices=quantized_coords.long())
            inds = torch.arange(num_points)

        if save_delta or save_delta_norm:
            pos_inds = data.pos[inds]
            grid_pos = grid_size * mapping_func(pos_inds / grid_size)
            delta = pos_inds - grid_pos
            if save_delta:
                new_data.delta = 2 * delta / grid_size

            if save_delta_norm:
                new_data.delta_norm = torch.norm(delta, p=2, dim=-1) / grid_size

        for key in set(data.keys):
            item = data[key].clone()
            if num_points == item.shape[0]:
                item = item[inds]
                setattr(new_data, key, item)

        if remove_duplicates:
            new_data = remove_duplicates_func(new_data)
        else:
            if apply_mean:
                new_data = apply_mean_func(new_data)
        return new_data
