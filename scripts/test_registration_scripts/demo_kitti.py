import open3d as o3d
import torch
import numpy as np
import os
import os.path as osp

import sys

sys.path.append("../..")

# data preprocessing import
from torch_points3d.core.data_transform import GridSampling3D, AddOnes, AddFeatByKey
from torch_geometric.transforms import Compose
from torch_geometric.data import Batch

# Model
from torch_points3d.applications.pretrained_api import PretainedRegistry

# post processing
from torch_points3d.utils.registration import get_matches, fast_global_registration


if __name__ == "__main__":
    # We read the data
    path_file = os.path.dirname(os.path.abspath(__file__))
    path_s = osp.join(path_file, "..", "..", "notebooks", "data", "KITTI", "000186.bin")
    path_t = osp.join(path_file, "..", "..", "notebooks", "data", "KITTI", "000200.bin")
    # path_t = "./notebooks/data/000049.bin"
    R_calib = np.asarray(
        [
            [-1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03],
            [-6.481465826011e-03, 8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02],
            [9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01],
        ]
    )
    pcd_s = np.fromfile(path_s, dtype=np.float32).reshape(-1, 4)[:, :3].dot(R_calib[:3, :3].T)
    pcd_t = np.fromfile(path_t, dtype=np.float32).reshape(-1, 4)[:, :3].dot(R_calib[:3, :3].T)

    transform = Compose(
        [
            GridSampling3D(mode="last", size=0.6, quantize_coords=True),
            AddOnes(),
            AddFeatByKey(add_to_x=True, feat_name="ones"),
        ]
    )

    data_s = transform(Batch(pos=torch.from_numpy(pcd_s).float(), batch=torch.zeros(pcd_s.shape[0]).long()))
    data_t = transform(Batch(pos=torch.from_numpy(pcd_t).float(), batch=torch.zeros(pcd_t.shape[0]).long()))

    model = PretainedRegistry.from_pretrained("minkowski-registration-kitti").cuda()

    with torch.no_grad():
        model.set_input(data_s, "cuda")
        output_s = model.forward()
        model.set_input(data_t, "cuda")
        output_t = model.forward()

    rand_s = torch.randint(0, len(output_s), (5000,))
    rand_t = torch.randint(0, len(output_t), (5000,))
    matches = get_matches(output_s[rand_s], output_t[rand_t])
    T_est = fast_global_registration(data_s.pos[rand_s][matches[:, 0]], data_t.pos[rand_t][matches[:, 1]])

    o3d_pcd_s = o3d.geometry.PointCloud()
    o3d_pcd_s.points = o3d.utility.Vector3dVector(data_s.pos.cpu().numpy())
    o3d_pcd_s.paint_uniform_color([0.9, 0.7, 0.1])

    o3d_pcd_t = o3d.geometry.PointCloud()
    o3d_pcd_t.points = o3d.utility.Vector3dVector(data_t.pos.cpu().numpy())
    o3d_pcd_t.paint_uniform_color([0.1, 0.7, 0.9])

    o3d.visualization.draw_geometries([o3d_pcd_s, o3d_pcd_t])
    o3d.visualization.draw_geometries([o3d_pcd_s.transform(T_est.cpu().numpy()), o3d_pcd_t])
