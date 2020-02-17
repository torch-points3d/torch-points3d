import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_accuracy(embedded_ref_features, embedded_val_features):

    """
    evaluate accuracy for a batch
    """
    number_of_test_points = embedded_ref_features.shape[0]
    neigh = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", metric="euclidean")
    neigh.fit(embedded_ref_features)
    dist_neigh_normal, ind_neigh_normal = neigh.kneighbors(embedded_val_features)
    reference_neighbors = np.reshape(np.arange(number_of_test_points), newshape=(-1, 1))

    wrong_matches = np.count_nonzero(ind_neigh_normal - reference_neighbors)
    accuracy = (1 - wrong_matches / number_of_test_points) * 100
    return accuracy
