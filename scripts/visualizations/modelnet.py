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

import open3d
import random
import numpy as np

from torch_points3d.datasets.classification.modelnet import SampledModelNet
import torch_points3d.core.data_transform as T3D
import torch_geometric.transforms as T


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


dataroot = os.path.join(DIR, "../data/modelnet")
pre_transform = T.Compose([T.NormalizeScale(), T3D.GridSampling3D(0.02)])
dataset = SampledModelNet(dataroot, name="40", train=True, transform=None, pre_transform=pre_transform, pre_filter=None)

colors = {}
while True:
    try:
        pcds = []
        for idx in range(25):
            i = np.random.randint(0, len(dataset))
            sample = dataset[i]
            label = sample.y.item()
            if label not in colors:
                color = np.asarray([np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)])
                colors[label] = color
            else:
                color = colors[label]
            pcd = torch2o3d(sample)
            pcd.paint_uniform_color(color)
            points = np.asarray(pcd.points) + np.tile(
                np.asarray([2 * idx, (idx * 2) % 5, 0])[np.newaxis, ...], (np.asarray(pcd.points).shape[0], 1)
            )
            pcd.points = open3d.utility.Vector3dVector(points)
            pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=70))
            pcds.append(pcd)
        open3d.visualization.draw_geometries(pcds)
    except KeyboardInterrupt:
        break
