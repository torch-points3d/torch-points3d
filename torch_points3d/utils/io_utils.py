import open3d


def torch2o3d(xyz, color=[1, 0, 0]):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
    pcd.paint_uniform_color(color)
    return pcd
