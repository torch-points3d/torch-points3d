import numpy as np
from sklearn.neighbors import NearestNeighbors


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


def compute_accuracy_gpu(embedded_ref_features, embedded_val_features):
    pass


def estimate_transfo(xyz, xyz_target, match):
    pass


def get_matches(feat, feat_target, num_matches=None):
    pass


def fast_global_registration(xyz, xyz_target, match):
    pass


def compute_hit_ratio(xyz, xyz_target, matches, T_gt, tau_1):
    pass


def compute_transfo_error(T_gt, T_pred):
    pass
