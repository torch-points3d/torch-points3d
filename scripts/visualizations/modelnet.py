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
import open3d

from torch_points3d.datasets.classification.modelnet import SampledModelNet
import torch_points3d.core.data_transform as T3D
import torch_geometric.transforms as T
from torch_points3d.utils.o3d_utils import *

dataroot = os.path.join(DIR, "../data/modelnet")
pre_transform = T.Compose([T.NormalizeScale(), T3D.GridSampling3D(0.02)])
dataset = SampledModelNet(dataroot, name="40", train=True, transform=None, pre_transform=pre_transform, pre_filter=None)

colors = {}
while True:
    try:
        pcds = []
        for idx in range(40):
            print(idx)
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
                np.asarray([4 * ((idx * 1) % 5), 3 * ((idx * 1) // 5), 0])[np.newaxis, ...],
                (np.asarray(pcd.points).shape[0], 1),
            )
            pcd.points = open3d.utility.Vector3dVector(points)
            pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=70))
            pcds.append(pcd)
        open3d.visualization.draw_geometries(pcds)
    except KeyboardInterrupt:
        break
