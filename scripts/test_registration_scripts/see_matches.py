"""
compute features get match and visualize match
only axcept fragment
"""

import open3d
import torch
import numpy as np
import hydra
import logging
from omegaconf import OmegaConf
import os
import sys
import pandas as pd
import time


DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..", "..")
sys.path.insert(0, ROOT)

from torch_points3d.models.model_factory import instantiate_model
from torch_points3d.datasets.dataset_factory import instantiate_dataset, get_dataset_class
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset

from torch_points3d.utils.registration import (
    estimate_transfo,
    teaser_pp_registration,
    fast_global_registration,
    get_matches,
)
from torch_points3d.metrics.registration_metrics import (
    compute_hit_ratio,
    compute_transfo_error,
    compute_scaled_registration_error,
)

from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

log = logging.getLogger(__name__)


def torch2o3d(data, color=None, ind=None):
    pcd = open3d.geometry.PointCloud()
    if ind is not None:
        pcd.points = open3d.utility.Vector3dVector(data.pos[ind].cpu().numpy())
    else:
        pcd.points = open3d.utility.Vector3dVector(data.pos.cpu().numpy())
    if color is not None:
        pcd.paint_uniform_color(color)
    pcd.estimate_normals()
    return pcd


def create_sphere(kp, color, radius):
    T = np.eye(4)
    T[:3, 3] = kp
    sphere = open3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.transform(T)
    sphere.paint_uniform_color(color)
    return sphere


def create_line(kp1, kp2, colors=np.array([[0, 0, 0]])):
    line_set = open3d.geometry.LineSet()
    points = [list(kp1), list(kp2)]

    lines = [[0, 1]]
    line_set.points = open3d.utility.Vector3dVector(points)
    line_set.lines = open3d.utility.Vector2iVector(lines)
    line_set.colors = open3d.utility.Vector3dVector(colors)
    return line_set


def match_visualizer(pcd1, keypoints1, pcd2, keypoints2, inliers, t=2, radius=0.1):
    """
    display the match using Open3D draw_geometries.
    input :
    -pcd: open3d point cloud
    -keypoints : the keypoints (open3d point cloud)
    -t: float a translation at the x axis
    condition : keypoints1 and keypoints2 have the same size
    """

    colors = []
    open3d.geometry.PointCloud(keypoints1)
    keypoints2_copy = open3d.geometry.PointCloud(keypoints2)
    T_trans = np.eye(4)
    T_trans[0, 3] = t
    list_displayed = [pcd1, pcd2]
    pcd2.transform(T_trans)

    keypoints2_copy.transform(T_trans)
    kp1 = np.asarray(keypoints1.points)
    kp2 = np.asarray(keypoints2_copy.points)

    if len(kp1) != len(kp2):
        raise Exception("number of points is different")
    for i in range(len(kp1)):
        assert inliers is not None
        col = inliers[i] * np.asarray([0, 1, 0]) + (1 - inliers[i]) * np.asarray([1, 0, 0])
        colors.append(list(col))
        p1 = kp1[i]
        p2 = kp2[i]
        sphere1 = create_sphere(p1, colors[-1], radius)
        sphere2 = create_sphere(p2, colors[-1], radius)
        line = create_line(p1, p2, col.reshape(1, 3))
        list_displayed.append(line)
        list_displayed.append(sphere1)
        list_displayed.append(sphere2)

    open3d.visualization.draw_geometries(list_displayed)


def run(model: BaseModel, dataset: BaseDataset, device, cfg):
    print(time.strftime("%Y%m%d-%H%M%S"))
    dataset.create_dataloaders(
        model, 1, False, cfg.training.num_workers, False,
    )
    loader = dataset.test_dataset[0]

    ind = 0
    if cfg.ind is not None:
        ind = cfg.ind
    t = 5
    if cfg.t is not None:
        t = cfg.t
    r = 0.1
    if cfg.r is not None:
        r = cfg.r
    print(loader)
    print(ind)
    data = loader[ind]
    data.batch = torch.zeros(len(data.pos)).long()
    data.batch_target = torch.zeros(len(data.pos_target)).long()
    print(data)
    with torch.no_grad():
        model.set_input(data, device)
        model.forward()

        name_scene, name_pair_source, name_pair_target = dataset.test_dataset[0].get_name(ind)
        print(name_scene, name_pair_source, name_pair_target)
        input, input_target = model.get_input()
        xyz, xyz_target = input.pos, input_target.pos
        ind, ind_target = input.ind, input_target.ind
        matches_gt = torch.stack([ind, ind_target]).transpose(0, 1)
        feat, feat_target = model.get_output()
        # rand = voxel_selection(xyz, grid_size=0.06, min_points=cfg.data.min_points)
        # rand_target = voxel_selection(xyz_target, grid_size=0.06, min_points=cfg.data.min_points)

        rand = torch.randperm(len(feat))[: cfg.data.num_points]
        rand_target = torch.randperm(len(feat_target))[: cfg.data.num_points]
        T_gt = estimate_transfo(xyz[matches_gt[:, 0]].clone(), xyz_target[matches_gt[:, 1]].clone())
        matches_pred = get_matches(feat[rand], feat_target[rand_target], sym=cfg.data.sym)
        # For color
        inliers = (
            torch.norm(
                xyz[rand][matches_pred[:, 0]] @ T_gt[:3, :3].T
                + T_gt[:3, 3]
                - xyz_target[rand_target][matches_pred[:, 1]],
                dim=1,
            )
            < cfg.data.tau_1
        )
        # compute transformation
        T_teaser = teaser_pp_registration(
            xyz[rand][matches_pred[:, 0]], xyz_target[rand_target][matches_pred[:, 1]], noise_bound=cfg.data.tau_1
        )
        pcd_source = torch2o3d(input, [1, 0.7, 0.1])

        pcd_target = torch2o3d(input_target, [0, 0.15, 0.9])
        open3d.visualization.draw_geometries([pcd_source, pcd_target])
        pcd_source.transform(T_teaser.cpu().numpy())
        open3d.visualization.draw_geometries([pcd_source, pcd_target])
        pcd_source.transform(np.linalg.inv(T_teaser.cpu().numpy()))
        rand_ind = torch.randperm(len(rand[matches_pred[:, 0]]))[:250]
        pcd_source.transform(T_gt.cpu().numpy())
        kp_s = torch2o3d(input, ind=rand[matches_pred[:, 0]][rand_ind])
        kp_s.transform(T_gt.cpu().numpy())
        kp_t = torch2o3d(input_target, ind=rand_target[matches_pred[:, 1]][rand_ind])
        match_visualizer(pcd_source, kp_s, pcd_target, kp_t, inliers[rand_ind].cpu().numpy(), radius=r, t=t)


@hydra.main(config_path="../../conf/config.yaml", strict=False)
def main(cfg):
    OmegaConf.set_struct(cfg, False)

    # Get device
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.training.cuda) else "cpu")
    log.info("DEVICE : {}".format(device))

    # Enable CUDNN BACKEND
    torch.backends.cudnn.enabled = cfg.training.enable_cudnn

    # Checkpoint
    checkpoint = ModelCheckpoint(cfg.training.checkpoint_dir, cfg.model_name, cfg.training.weight_name, strict=True)

    # Setup the dataset config
    # Generic config

    dataset = instantiate_dataset(cfg.data)
    model = checkpoint.create_model(dataset, weight_name=cfg.training.weight_name)
    log.info(model)
    log.info("Model size = %i", sum(param.numel() for param in model.parameters() if param.requires_grad))

    log.info(dataset)

    model.eval()
    if cfg.enable_dropout:
        model.enable_dropout_in_eval()
    model = model.to(device)

    run(model, dataset, device, cfg)


if __name__ == "__main__":
    main()
