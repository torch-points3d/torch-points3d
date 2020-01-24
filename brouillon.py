import open3d
import os
import os.path as osp
import torch
import numpy as np


def torch2o3d(x):

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(x.detach().numpy())
    return pcd


def visu_pcd():
    # root = '../data/fragment/train_tiny/sun3d-mit_32_d507-d507_2'
    root = "../data/fragment/train_tiny/7-scenes-chess"
    for i in range(20):
        x = torch.load(osp.join(root, "fragment_{}.pt".format(i)))
        pcd = torch2o3d(x)
        downpcd = pcd.voxel_down_sample(voxel_size=0.05)
        downpcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=17))
        downpcd.paint_uniform_color([0.5, 0.5, 0.5])
        open3d.visualization.draw_geometries([downpcd])


if __name__ == "__main__":

    visu_pcd()
