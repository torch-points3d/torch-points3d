import os
import sys
import importlib

DIR = os.path.dirname(os.path.realpath(__file__))
torch_points3d = os.path.join(DIR, "..", "..", "torch_points3d")
assert os.path.exists(torch_points3d)

MODULE_PATH = os.path.join(torch_points3d, "__init__.py")
MODULE_NAME = "torch_points3d"
spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

from omegaconf import OmegaConf
import numpy as np
import torch
import open3d

from torch_points3d.datasets.panoptic.s3dis import S3DISFusedDataset
from torch_points3d.utils.o3d_utils import *


dataset_options = OmegaConf.load(os.path.join(DIR, "../../conf/data/panoptic/s3disfused.yaml"))

dataset_options.data.dataroot = os.path.join(DIR, "../../data")
dataset = S3DISFusedDataset(dataset_options.data)
print(dataset)

dataset._train_dataset.transform = None

while True:
    try:
        i = np.random.randint(0, len(dataset.train_dataset))
        sample = dataset.train_dataset[i]
        pcd = torch2o3d(sample)
        pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=70))
        list_objects = []
        existing_colors = []
        mask = sample.instance_mask
        scene = apply_mask(sample, torch.logical_not(mask))
        scene_pcd = torch2o3d(scene, color=[0.8, 0.8, 0.8])

        scene_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=70))
        scene_pcd = scene_pcd.voxel_down_sample(0.07)

        for i in range(1, sample.num_instances.item() + 1):
            instance_mask = sample.instance_labels == i
            obj = apply_mask(sample, instance_mask)
            new_color = generate_new_color(existing_colors)
            pcd = torch2o3d(obj, color=new_color)
            existing_colors.append(new_color)
            pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=70))
            list_objects.append(pcd)

        print()
        print(sample)
        open3d.visualization.draw_geometries([scene_pcd, *list_objects])
    except KeyboardInterrupt:
        break
