import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


# Metrics utilities


def compute_accuracy(embedded_ref_features, embedded_val_features):

    """
    accuracy for metric learning tasks in case descriptor learning
    Args:
        embedded_ref_feature(numpy array): size N x D the features computed by one head of the siamese
        embedded_val_features(array): N x D the features computed by the other head
    return:
        each line of the matrix are supposed to be equal, this function counts when it's the case
    """
    number_of_test_points = embedded_ref_features.shape[0]
    neigh = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", metric="euclidean")
    neigh.fit(embedded_ref_features)
    dist_neigh_normal, ind_neigh_normal = neigh.kneighbors(embedded_val_features)
    reference_neighbors = np.reshape(np.arange(number_of_test_points), newshape=(-1, 1))

    wrong_matches = np.count_nonzero(ind_neigh_normal - reference_neighbors)
    accuracy = (1 - wrong_matches / number_of_test_points) * 100
    return accuracy


def compute_hit_ratio(xyz, xyz_target, T_gt, tau_1):
    """
    compute proportion of point which are close.
    """
    assert xyz.shape == xyz.shape
    dist = torch.norm(xyz.mm(T_gt[:3, :3].T) + T_gt[:3, 3] - xyz_target, dim=1)

    return torch.mean((dist < tau_1).to(torch.float))


def compute_transfo_error(T_gt, T_pred):
    """
    compute the translation error (the unit depends on the unit of the point cloud)
    and compute the rotation error in degree using the formula (norm of antisymetr):
    http://jlyang.org/tpami16_go-icp_preprint.pdf
    """
    rte = torch.norm(T_gt[:3, 3] - T_pred[:3, 3])
    cos_theta = (torch.trace(T_gt[:3, :3].mm(T_pred[:3, :3].T)) - 1) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    rre = torch.acos(cos_theta) * 180 / np.pi
    return rte, rre
