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

import panel as pn
import numpy as np
import pyvista as pv

pv.set_plot_theme("document")
from matplotlib.colors import ListedColormap
from omegaconf import OmegaConf
import random

pn.extension("vtk")
os.system("/usr/bin/Xvfb :99 -screen 0 1024x768x24 &")
os.environ["DISPLAY"] = ":99"
os.environ["PYVISTA_OFF_SCREEN"] = "True"
os.environ["PYVISTA_USE_PANEL"] = "True"

DIR = os.path.dirname(os.getcwd())
sys.path.append(DIR)

import torch
import open3d
import random
import numpy as np

from torch_points3d.datasets.panoptic.scannet import ScannetDataset, ScannetPanoptic
from torch_points3d.datasets.segmentation.scannet import Scannet, SCANNET_COLOR_MAP
from torch_points3d.datasets.segmentation import IGNORE_LABEL


def get_random_color(pastel_factor=0.5):
    return [(x + pastel_factor) / (1.0 + pastel_factor) for x in [random.uniform(0, 1.0) for i in [1, 2, 3]]]


def color_distance(c1, c2):
    return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])


def generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor=pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color


def torch2o3d(data, color=[1, 0, 0]):
    xyz = data.pos
    norm = data.norm
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
    if norm is not None:
        pcd.normals = open3d.utility.Vector3dVector(norm.detach().cpu().numpy())
    pcd.paint_uniform_color(color)
    return pcd


def apply_mask(d, mask, skip_keys=[]):
    data = d.clone()
    size_pos = len(data.pos)
    for k in data.keys:
        if size_pos == len(data[k]) and k not in skip_keys:
            data[k] = data[k][mask]
    return data


dataset_options = OmegaConf.load(os.path.join(DIR, "../conf/data/panoptic/scannet-sparse.yaml"))

dataset_options.data.dataroot = os.path.join(DIR, "../data")
dataset = ScannetDataset(dataset_options.data)
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
