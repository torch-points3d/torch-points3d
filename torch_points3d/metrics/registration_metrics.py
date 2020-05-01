import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import knn


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


def estimate_transfo(xyz, xyz_target):
    """
    estimate the rotation and translation using Kabsch algorithm
    Parameters:
    xyz :
    xyz_target:
    """
    assert xyz.shape == xyz.shape
    xyz_c = xyz - xyz.mean(0)
    xyz_target_c = xyz_target - xyz_target.mean(0)
    Q = xyz_c.T.mm(xyz_target_c) / len(xyz)
    U, S, V = torch.svd(Q)
    d = torch.det(V.mm(U.T))
    diag = torch.diag(torch.tensor([1, 1, d], device=xyz.device))
    R = V.mm(diag).mm(U.T)
    t = xyz_target.mean(0) - R @ xyz.mean(0)
    T = torch.eye(4, device=xyz.device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def get_geman_mclure_weight(xyz, xyz_target, mu):
    """
    compute the weights defined here for the iterative reweighted least square.
    http://vladlen.info/papers/fast-global-registration.pdf
    """
    norm2 = torch.norm(xyz_target - xyz, dim=1) ** 2
    return (mu / (mu + norm2)).view(-1, 1)


def get_cross_product_matrix(k):
    return torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], device=k.device)


def rodrigues(axis, theta):
    """
    given an axis of norm one and an angle, compute the rotation matrix using rodrigues formula
    source : https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    K = get_cross_product_matrix(axis)
    t = torch.tensor([theta], device=axis.device)
    R = torch.eye(3, device=axis.device) + torch.sin(t) * K + (1 - torch.cos(t)) * K.mm(K)
    return R


def get_matrix_system(xyz, xyz_target, weight):
    """
    Build matrix of size 3N x 6 and b of size 3N

    xyz size N x 3
    xyz_target size N x 3
    weight size N
    the matrix is minus cross product matrix concatenate with the identity (rearanged).
    """
    assert xyz.shape == xyz_target.shape
    A_x = torch.zeros(xyz.shape[0], 6, device=xyz.device)
    A_y = torch.zeros(xyz.shape[0], 6, device=xyz.device)
    A_z = torch.zeros(xyz.shape[0], 6, device=xyz.device)
    b_x = weight.view(-1) * (xyz_target[:, 0] - xyz[:, 0])
    b_y = weight.view(-1) * (xyz_target[:, 1] - xyz[:, 1])
    b_z = weight.view(-1) * (xyz_target[:, 2] - xyz[:, 2])
    A_x[:, 1] = weight.view(-1) * xyz[:, 2]
    A_x[:, 2] = -weight.view(-1) * xyz[:, 1]
    A_x[:, 3] = weight.view(-1) * 1
    A_y[:, 0] = -weight.view(-1) * xyz[:, 2]
    A_y[:, 2] = weight.view(-1) * xyz[:, 0]
    A_y[:, 4] = weight.view(-1) * 1
    A_z[:, 0] = weight.view(-1) * xyz[:, 1]
    A_z[:, 1] = -weight.view(-1) * xyz[:, 0]
    A_z[:, 5] = weight.view(-1) * 1
    return torch.cat([A_x, A_y, A_z], 0), torch.cat([b_x, b_y, b_z], 0).view(-1, 1)


def get_trans(x):
    """
    get the matrix
    """
    T = torch.eye(4, device=x.device)
    T[:3, 3] = x[3:]
    axis = x[:3]
    theta = torch.norm(axis)
    if theta > 0:
        axis = axis / theta
    T[:3, :3] = rodrigues(axis, theta)
    return T


def fast_global_registration(xyz, xyz_target, mu_init=1, num_iter=20):
    """
    estimate the rotation and translation using Fast Global Registration algorithm (M estimator for robust estimation)
    http://vladlen.info/papers/fast-global-registration.pdf
    """
    assert xyz.shape == xyz_target.shape

    T_res = torch.eye(4, device=xyz.device)
    mu = mu_init
    source = xyz.clone()
    weight = torch.ones(len(source), 1, device=xyz.device)
    for i in range(num_iter):
        if i > 0 and i % 5 == 0:
            mu /= 2.0
        A, b = get_matrix_system(source, xyz_target, weight)
        x, _ = torch.solve(A.T @ b, A.T.mm(A))
        T = get_trans(x.view(-1))
        source = source.mm(T[:3, :3].T) + T[:3, 3]
        T_res = T @ T_res
        weight = get_geman_mclure_weight(source, xyz_target, mu)
    return T_res


def compute_hit_ratio(xyz, xyz_target, T_gt, tau_1):
    """
    compute proportion of point which are close.
    """
    assert xyz.shape == xyz.shape
    dist = torch.norm(xyz.mm(T_gt[:3, :3].T) + T_gt[:3, 3] - xyz_target, dim=1)

    return torch.mean((dist < 0.1).to(torch.float))


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


def get_matches(feat_source, feat_target, sym=False):

    matches = knn(feat_target, feat_source, k=1).T
    if sym:
        match_inv = knn(feat_source, feat_target, k=1).T
        mask = match_inv[matches[:, 1], 1] == torch.arange(matches.shape[0])
        return matches[mask]
    else:
        return matches
