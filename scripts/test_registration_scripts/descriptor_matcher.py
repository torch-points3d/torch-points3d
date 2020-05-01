import pandas as pd
import hydra
import numpy as np
from sklearn.neighbors import KDTree
import os
import os.path as osp
from omegaconf import OmegaConf
import sys
import matplotlib.pyplot as plt

# Import building function for model and dataset
DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..", "..")
sys.path.insert(0, ROOT)


def read_gt_log(path):
    """
    read the gt.log of evaluation set of 3DMatch or ETH Dataset and parse it.
    """
    list_pair = []
    list_mat = []
    with open(path, "r") as f:
        all_mat = f.readlines()
    mat = np.zeros((4, 4))
    for i in range(len(all_mat)):
        if i % 5 == 0:
            if i != 0:
                list_mat.append(mat)
            mat = np.zeros((4, 4))
            list_pair.append(list(map(int, all_mat[i].split("\t")[:-1])))
        else:
            line = all_mat[i].split("\t")

            mat[i % 5 - 1] = np.asarray(line[:4], dtype=np.float)
    list_mat.append(mat)
    return list_pair, list_mat


def compute_matches(feature_source, feature_target, kp_source, kp_target, ratio=False, sym=False):
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
        indmatch = np.arange(len(feature_target))
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

    feat_s = data_source["feat"]
    feat_t = data_target["feat"]

    if len(data_source["feat"]) != len(data_source["keypoints"]):
        # Sampled features using keypoints.
        feat_s = feat_s[data_source["keypoints"]]
        feat_t = feat_t[data_target["keypoints"]]

    kp_source, kp_target = compute_matches(
        feat_s, feat_t, data_source["pcd"][data_source["keypoints"]], data_target["pcd"][data_target["keypoints"]],
    )

    dist = compute_dists(kp_source, kp_target, gt_trans)

    n_s = osp.split(path_descr_source)[-1].split(".")[0]
    n_t = osp.split(path_descr_target)[-1].split(".")[0]

    frac_correct = compute_mean_correct_matches(dist, list_tau)

    dico = dict(
        kp_source=kp_source,
        kp_target=kp_target,
        dist=dist,
        list_tau1=list_tau,
        frac_correct=frac_correct,
        name_source=n_s,
        name_target=n_t,
    )
    print(n_s, n_t, frac_correct)
    return dico


def compute_recall_scene(scene_name, list_pair, list_trans, list_tau1, list_tau2, res_path):
    """
    evaluate the recall for each scene
    """
    list_frac_correct = []
    list_dico = []
    for i, pair in enumerate(list_pair):
        dico = pair_evaluation(pair[0], pair[1], list_trans[i], list_tau1, res_path)
        list_frac_correct.append(dico["frac_correct"])
        list_dico.append(dico)

    list_recall = compute_mean_correct_matches(np.asarray(list_frac_correct), list_tau2, is_leq=False)
    print("Save the matches")
    df_matches = pd.DataFrame(list_dico)
    df_matches.to_csv(osp.join(res_path, "matches.csv"))
    dico = dict(scene_name=scene_name, list_tau2=list(list_tau2), list_recall=list(list_recall))
    df_res = pd.DataFrame([dico])
    df_res.to_csv(osp.join(res_path, "res_recall.csv"))
    return dico


def evaluate(path_raw_fragment, path_results, list_tau1, list_tau2):

    """
    launch the evaluation procedure
    """

    list_scene = os.listdir(path_raw_fragment)
    list_total_res = []
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
        dico = compute_recall_scene(scene, list_pair, list_mat, list_tau1, list_tau2, res_path)
        list_total_res.append(dico)
    total_recall = np.mean([d["list_recall"] for d in list_total_res], axis=0)
    list_total_res.append(dict(scene_name="total", list_tau2=list_tau2, list_recall=list(total_recall)))
    df = pd.DataFrame(list_total_res)
    df.to_csv(osp.join(path_results, "matches", "total_res.csv"))


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    print(cfg)
    evaluate(cfg.path_raw_fragment, cfg.path_results, cfg.list_tau1, cfg.list_tau2)


if __name__ == "__main__":
    main()
