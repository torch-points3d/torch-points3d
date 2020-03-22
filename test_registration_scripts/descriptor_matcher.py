import hydra
import numpy as np
from sklearn.neighbors import KDTree
import os
import os.path as osp
import json
from omegaconf import OmegaConf
import sys

# Import building function for model and dataset
DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

from src.datasets.registration.evaluation import read_gt_log


def compute_matches(feature_source, feature_target, kp_source, kp_target, ratio=False, sym=True):
    """
    compute matches between features
    """
    tree_source = KDTree(feature_source)
    tree_target = KDTree(feature_target)
    _, nn_ind_source = tree_target.query(feature_source, k=1)
    _, nn_ind_target = tree_source.query(feature_target, k=1)

    # symetric test
    if sym:
        indmatch = np.where(nn_ind_source.T[0][nn_ind_target.T[0]] == np.arange(len(feature_source)))[0]
    else:
        indmatch = np.arange(len(feature_source))

    new_kp_source = np.copy(kp_source[nn_ind_target.T[0][indmatch]])
    new_kp_target = np.copy(kp_target[indmatch])

    return new_kp_source, new_kp_target


def compute_dists(kp_source, kp_target, trans):
    """
    compute distance between points that are matches
    """
    kp_target_t = kp_target.dot(trans[:3, :3].T) + trans[:3, 3]
    dist = np.linalg.norm(kp_source - kp_target_t, axis=1)
    return dist


def compute_mean_correct_matches(dist, list_tau, is_leq=True):
    """
    for each pair, save the name of the source,
    """
    res = []
    for tau in list_tau:
        if is_leq:
            res.append(np.mean(dist < tau))
        else:
            res.append(np.mean(dist > tau))
    return res


def pair_evaluation(path_descr_source, path_descr_target, gt_trans, list_tau, res_path):
    """
    save matches (indices)
    """

    data_source = np.load(path_descr_source)
    data_target = np.load(path_descr_target)

    kp_source, kp_target = compute_matches(
        data_source["feat"],
        data_target["feat"],
        data_source["pcd"][data_source["keypoints"]],
        data_target["pcd"][data_target["keypoints"]],
    )

    dist = compute_dists(kp_source, kp_target, gt_trans)

    n_s = osp.split(path_descr_source)[-1].split(".")[0]
    n_t = osp.split(path_descr_target)[-1].split(".")[0]
    out_path = osp.join(res_path, "{}_{}.npz".format(n_s, n_t))
    frac_correct = compute_mean_correct_matches(dist, list_tau)
    print(n_s, n_t, frac_correct[0:5], list_tau)
    np.savez(
        out_path,
        kp_source,
        kp_target,
        dist=dist,
        list_tau1=list_tau,
        frac_correct=frac_correct,
        name_source=n_s,
        name_target=n_t,
    )
    return frac_correct[0]


def compute_recall_scene(scene_name, list_pair, list_trans, list_tau1, list_tau2, res_path):
    """
    evaluate the recall for each scene
    """
    list_frac_correct = []
    for i, pair in enumerate(list_pair):
        u_st = pair_evaluation(pair[0], pair[1], list_trans[i], list_tau1, res_path)
        list_frac_correct.append(u_st)

    list_recall = compute_mean_correct_matches(np.asarray(list_frac_correct), list_tau2, is_leq=False)
    out_path = osp.join(res_path, "res_recall.json")
    with open(out_path, "w") as f:
        print(list_recall)
        print(list_tau2)
        print(scene_name)
        dico = dict(scene_name=scene_name, list_tau2=list(list_tau2), list_recall=list(list_recall))
        json.dump(dico, f)


def evaluate(path_raw_fragment, path_results, list_tau1, list_tau2):

    """
    launch the evaluation procedure
    """

    list_scene = os.listdir(path_raw_fragment)

    for scene in list_scene:
        print(scene)
        path_log = osp.join(path_raw_fragment, scene, "gt.log")
        list_pair_num, list_mat = read_gt_log(path_log)
        list_pair = []
        for pair in list_pair_num:
            name0 = "{}_{}_desc.npz".format("cloud_bin", pair[0])
            name1 = "{}_{}_desc.npz".format("cloud_bin", pair[1])
            list_pair.append(
                [osp.join(path_results, "features", scene, name0), osp.join(path_results, "features", scene, name1)]
            )
        res_path = osp.join(path_results, "matches", scene)
        if not osp.exists(res_path):
            os.makedirs(res_path, exist_ok=True)
        compute_recall_scene(scene, list_pair, list_mat, list_tau1, list_tau2, res_path)


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    print(cfg)
    evaluate(cfg.path_raw_fragment, cfg.path_results, cfg.list_tau1, cfg.list_tau2)


if __name__ == "__main__":
    main()
