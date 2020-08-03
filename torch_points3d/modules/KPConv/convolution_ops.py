# defining KPConv using torch ops
# Adaptation of https://github.com/HuguesTHOMAS/KPConv/
# Adaption from https://github.com/humanpose1/KPConvTorch/

import torch
from torch_points3d.core.common_modules.gathering import gather


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * sig ** 2 + eps))


def KPConv_ops(
    query_points,
    support_points,
    neighbors_indices,
    features,
    K_points,
    K_values,
    KP_extent,
    KP_influence,
    aggregation_mode,
):
    """
    This function creates a graph of operations to define Kernel Point Convolution in tensorflow. See KPConv function
    above for a description of each parameter
    :param query_points: float32[n_points, dim] - input query points (center of neighborhoods)
    :param support_points: float32[n0_points, dim] - input support points (from which neighbors are taken)
    :param neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
    :param features: float32[n0_points, in_fdim] - input features
    :param K_values: float32[n_kpoints, in_fdim, out_fdim] - weights of the kernel
    :param fixed: string in ('none', 'center' or 'verticals') - fix position of certain kernel points
    :param KP_extent: float32 - influence radius of each kernel point
    :param KP_influence: string in ('constant', 'linear', 'gaussian') - influence function of the kernel points
    :param aggregation_mode: string in ('closest', 'sum') - whether to sum influences, or only keep the closest
    :return:                    [n_points, out_fdim]
    """

    # Get variables
    int(K_points.shape[0])

    # Add a fake point in the last row for shadow neighbors
    shadow_point = torch.ones_like(support_points[:1, :]) * 1e6
    support_points = torch.cat([support_points, shadow_point], dim=0)

    # Get neighbor points [n_points, n_neighbors, dim]
    neighbors = gather(support_points, neighbors_indices)

    # Center every neighborhood
    neighbors = neighbors - query_points.unsqueeze(1)

    # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
    neighbors.unsqueeze_(2)
    differences = neighbors - K_points

    # Get the square distances [n_points, n_neighbors, n_kpoints]
    sq_distances = torch.sum(differences ** 2, dim=3)

    # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
    if KP_influence == "constant":
        # Every point get an influence of 1.
        all_weights = torch.ones_like(sq_distances)
        all_weights = all_weights.transpose(2, 1)

    elif KP_influence == "linear":
        # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
        all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / KP_extent, min=0.0)
        all_weights = all_weights.transpose(2, 1)

    elif KP_influence == "gaussian":
        # Influence in gaussian of the distance.
        sigma = KP_extent * 0.3
        all_weights = radius_gaussian(sq_distances, sigma)
        all_weights = all_weights.transpose(2, 1)
    else:
        raise ValueError("Unknown influence function type (config.KP_influence)")

    # In case of closest mode, only the closest KP can influence each point
    if aggregation_mode == "closest":
        neighbors_1nn = torch.argmin(sq_distances, dim=-1)
        all_weights *= torch.transpose(torch.nn.functional.one_hot(neighbors_1nn, K_points.shape[0]), 1, 2)

    elif aggregation_mode != "sum":
        raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

    features = torch.cat([features, torch.zeros_like(features[:1, :])], dim=0)

    # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
    neighborhood_features = gather(features, neighbors_indices)

    # Apply distance weights [n_points, n_kpoints, in_fdim]
    weighted_features = torch.matmul(all_weights, neighborhood_features)

    # Apply network weights [n_kpoints, n_points, out_fdim]
    weighted_features = weighted_features.permute(1, 0, 2)
    kernel_outputs = torch.matmul(weighted_features, K_values)

    # Convolution sum to get [n_points, out_fdim]
    output_features = torch.sum(kernel_outputs, dim=0)

    return output_features


def KPConv_deform_ops(
    query_points,
    support_points,
    neighbors_indices,
    features,
    K_points,
    offsets,
    modulations,
    K_values,
    KP_extent,
    KP_influence,
    aggregation_mode,
):
    """
    This function creates a graph of operations to define Deformable Kernel Point Convolution in tensorflow. See
    KPConv_deformable function above for a description of each parameter
    :param query_points:        [n_points, dim]
    :param support_points:      [n0_points, dim]
    :param neighbors_indices:   [n_points, n_neighbors]
    :param features:            [n0_points, in_fdim]
    :param K_points:            [n_kpoints, dim]
    :param offsets:             [n_points, n_kpoints, dim]
    :param modulations:         [n_points, n_kpoints] or None
    :param K_values:            [n_kpoints, in_fdim, out_fdim]
    :param KP_extent:           float32
    :param KP_influence:        string
    :param aggregation_mode:    string in ('closest', 'sum') - whether to sum influences, or only keep the closest

    :return features, square_distances, deformed_K_points
    """

    # Get variables
    n_kp = int(K_points.shape[0])
    shadow_ind = support_points.shape[0]

    # Add a fake point in the last row for shadow neighbors
    shadow_point = torch.ones_like(support_points[:1, :]) * 1e6
    support_points = torch.cat([support_points, shadow_point], axis=0)

    # Get neighbor points [n_points, n_neighbors, dim]
    neighbors = support_points[neighbors_indices]

    # Center every neighborhood
    neighbors = neighbors - query_points.unsqueeze(1)

    # Apply offsets to kernel points [n_points, n_kpoints, dim]
    deformed_K_points = torch.add(offsets, K_points)

    # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
    neighbors = neighbors.unsqueeze(2)
    neighbors = neighbors.repeat([1, 1, n_kp, 1])
    differences = neighbors - deformed_K_points.unsqueeze(1)

    # Get the square distances [n_points, n_neighbors, n_kpoints]
    sq_distances = torch.sum(differences ** 2, axis=3)

    # Boolean of the neighbors in range of a kernel point [n_points, n_neighbors]
    in_range = (sq_distances < KP_extent ** 2).any(2).to(torch.long)

    # New value of max neighbors
    new_max_neighb = torch.max(torch.sum(in_range, axis=1))
    # print(new_max_neighb)

    # For each row of neighbors, indices of the ones that are in range [n_points, new_max_neighb]
    new_neighb_bool, new_neighb_inds = torch.topk(in_range, k=new_max_neighb)

    # Gather new neighbor indices [n_points, new_max_neighb]
    new_neighbors_indices = neighbors_indices.gather(1, new_neighb_inds)

    # Gather new distances to KP [n_points, new_max_neighb, n_kpoints]
    new_neighb_inds_sq = new_neighb_inds.unsqueeze(-1)
    new_sq_distances = sq_distances.gather(1, new_neighb_inds_sq.repeat((1, 1, sq_distances.shape[-1])))

    # New shadow neighbors have to point to the last shadow point
    new_neighbors_indices *= new_neighb_bool
    new_neighbors_indices += (1 - new_neighb_bool) * shadow_ind

    # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
    if KP_influence == "constant":
        # Every point get an influence of 1.
        all_weights = (new_sq_distances < KP_extent ** 2).to(torch.float32)
        all_weights = all_weights.permute(0, 2, 1)

    elif KP_influence == "linear":
        # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
        all_weights = torch.relu(1 - torch.sqrt(new_sq_distances) / KP_extent)
        all_weights = all_weights.permute(0, 2, 1)

    elif KP_influence == "gaussian":
        # Influence in gaussian of the distance.
        sigma = KP_extent * 0.3
        all_weights = radius_gaussian(new_sq_distances, sigma)
        all_weights = all_weights.permute(0, 2, 1)
    else:
        raise ValueError("Unknown influence function type (config.KP_influence)")

    # In case of closest mode, only the closest KP can influence each point
    if aggregation_mode == "closest":
        neighbors_1nn = torch.argmin(new_sq_distances, axis=2, output_type=torch.long)
        all_weights *= torch.zeros_like(all_weights, dtype=torch.float32).scatter_(1, neighbors_1nn, 1)

    elif aggregation_mode != "sum":
        raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

    features = torch.cat([features, torch.zeros_like(features[:1, :])], axis=0)

    # Get the features of each neighborhood [n_points, new_max_neighb, in_fdim]
    neighborhood_features = features[new_neighbors_indices]

    # Apply distance weights [n_points, n_kpoints, in_fdim]
    # print(all_weights.shape, neighborhood_features.shape)
    weighted_features = torch.matmul(all_weights, neighborhood_features)

    # Apply modulations
    if modulations is not None:
        weighted_features *= modulations.unsqueeze(2)

    # Apply network weights [n_kpoints, n_points, out_fdim]
    weighted_features = weighted_features.permute(1, 0, 2)
    kernel_outputs = torch.matmul(weighted_features, K_values)

    # Convolution sum [n_points, out_fdim]
    output_features = torch.sum(kernel_outputs, axis=0)

    # we need regularization
    return output_features, sq_distances, deformed_K_points
