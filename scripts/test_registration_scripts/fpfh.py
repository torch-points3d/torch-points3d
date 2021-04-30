import open3d
import torch
import numpy as np
import hydra
import os
import os.path as osp
import sys
import json
from omegaconf import OmegaConf

# Import building function for model and dataset
DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

from test_registration_scripts.save_feature import save


class FPFH(object):
    def __init__(self, radius=0.3, max_nn=128, radius_normal=0.3, max_nn_normal=17):
        """
        given a fragment, compute FPFH descriptor for keypoints
        """
        self.kdtree = open3d.geometry.KDTreeSearchParamHybrid(radius, max_nn)
        self.kdtree_normal = open3d.geometry.KDTreeSearchParamHybrid(radius_normal, max_nn_normal)

    def __call__(self, data):
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(data.pos.numpy())
        pcd.estimate_normals(self.kdtree_normal)
        fpfh_feature = open3d.pipelines.registration.compute_fpfh_feature(pcd, self.kdtree)
        return np.asarray(fpfh_feature.data).T[data.keypoints.numpy()]


@hydra.main(config_path="conf/fpfh.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    print(cfg)
    input_path = cfg.input_path
    output_path = cfg.output_path
    radius = cfg.radius
    max_nn = cfg.max_nn
    radius_normal = cfg.radius_normal
    max_nn_normal = cfg.max_nn_normal

    fpfh = FPFH(radius, max_nn, radius_normal, max_nn_normal)

    list_frag = sorted([f for f in os.listdir(input_path) if "fragment" in f])
    path_table = osp.join(input_path, "table.json")
    with open(path_table, "r") as f:
        table = json.load(f)

    for i in range(len(list_frag)):
        print(i, table[str(i)], list_frag[i])
        data = torch.load(osp.join(input_path, list_frag[i]))
        feat = fpfh(data)
        save(osp.join(output_path, "features"), table[str(i)]["scene_path"], table[str(i)]["fragment_name"], data, feat)


if __name__ == "__main__":
    main()
