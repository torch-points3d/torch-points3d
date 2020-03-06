import numpy as np
import torch
import MinkowskiEngine as ME
from torch_geometric.data import Data


def to_sparse_input(data, grid_size, save_delta=True, save_delta_norm=True, remove_duplicates=True):
    if grid_size is None:
        raise Exception("Grid size should be provided")

    elif grid_size == 0:
        raise Exception("Grid size should not be equal to 0")

    else:
        num_points = data.pos.shape[0]
        quantized_coords = np.rint(data.pos / grid_size)

        inds = ME.utils.sparse_quantize(quantized_coords, return_index=True)

        indices = quantized_coords[inds]

        new_data = Data(indices=indices)
        if save_delta or save_delta_norm:
            pos_inds = data.pos[inds]
            grid_pos = grid_size * np.rint(pos_inds / grid_size)
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

        return new_data
