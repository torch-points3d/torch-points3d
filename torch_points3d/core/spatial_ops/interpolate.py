import torch
from torch_geometric.nn import knn_interpolate, knn
from torch_scatter import scatter_add
from torch_geometric.data import Data


class KNNInterpolate:
    def __init__(self, k):
        self.k = k

    def precompute(self, query, support):
        """ Precomputes a data structure that can be used in the transform itself to speed things up
        """
        pos_x, pos_y = query.pos, support.pos
        if hasattr(support, "batch"):
            batch_y = support.batch
        else:
            batch_y = torch.zeros((support.num_nodes,), dtype=torch.long)
        if hasattr(query, "batch"):
            batch_x = query.batch
        else:
            batch_x = torch.zeros((query.num_nodes,), dtype=torch.long)

        with torch.no_grad():
            assign_index = knn(pos_x, pos_y, self.k, batch_x=batch_x, batch_y=batch_y)
            y_idx, x_idx = assign_index
            diff = pos_x[x_idx] - pos_y[y_idx]
            squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
            weights = 1.0 / torch.clamp(squared_distance, min=1e-16)
        normalisation = scatter_add(weights, y_idx, dim=0, dim_size=pos_y.size(0))

        return Data(num_nodes=support.num_nodes, x_idx=x_idx, y_idx=y_idx, weights=weights, normalisation=normalisation)

    def __call__(self, query, support, precomputed: Data = None):
        """ Computes a new set of features going from the query resolution position to the support
        resolution position
        Args:
            - query: data structure that holds the low res data (position + features)
            - support: data structure that holds the position to which we will interpolate
        Returns:
            - torch.tensor: interpolated features
        """
        if precomputed:
            num_points = support.pos.size(0)
            if num_points != precomputed.num_nodes:
                raise ValueError("Precomputed indices do not match with the data given to the transform")

            x = query.x
            x_idx, y_idx, weights, normalisation = (
                precomputed.x_idx,
                precomputed.y_idx,
                precomputed.weights,
                precomputed.normalisation,
            )
            y = scatter_add(x[x_idx] * weights, y_idx, dim=0, dim_size=num_points)
            y = y / normalisation
            return y

        x, pos = query.x, query.pos
        pos_support = support.pos
        if hasattr(support, "batch"):
            batch_support = support.batch
        else:
            batch_support = torch.zeros((support.num_nodes,), dtype=torch.long)
        if hasattr(query, "batch"):
            batch = query.batch
        else:
            batch = torch.zeros((query.num_nodes,), dtype=torch.long)

        return knn_interpolate(x, pos, pos_support, batch, batch_support, k=self.k)
