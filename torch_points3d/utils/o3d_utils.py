import open3d
import random


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
