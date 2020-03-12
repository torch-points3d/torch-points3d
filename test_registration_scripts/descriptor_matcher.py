import numpy as np
from sklearn.neighbors import KDTree


def compute_matches(feat_source, feat_target):
    """
    compute matches between features(list of tuple of indices)
    and return dist
    """
    tree = KDTree(feat_target)
    ind_source = np.arange(0, len(feat_source["feat"])).reshape(-1, 1)
    dist, ind_target = tree.query(feat_source["feat"], k=1)
    inds = np.hstack((ind_source, ind_target))
    return inds


def sym_test(inds):
    pass


def ratio_test(inds):
    pass


def compute_dists(pcd_source, pcd_target, trans, inds):
    """
    compute distance between points that are matches
    """
    pcd_source_t = pcd_source.dot(trans[:3, :3].T) + trans[:3, 3]
    dist = np.linalg.norm(pcd_source_t[inds[:, 0]] - pcd_target[inds[:, 1]], axis=1)
    return dist


def save_matches(inds, dist, out_path):
    """
    for each pair, save the name of the source,
    """
    dico = dict()
    dico["inds"] = inds
    dico["dist"] = dist

    np.save(out_path, dico)


def pair_evaluation(path_descr_source, path_descr_target):
    """
    save matches (indices)
    """

    data_source = np.load(path_descr_source)
    data_target = np.load(path_descr_target)

    gt_trans = ...

    inds = compute_matches(data_source["feat"], data_target["feat"])
    dist = compute_dists(data_source["pcd"], data_target["pcd"], gt_trans, inds)

    out = ...
    save_matches(inds, dist, out)
