import collections
import errno
import numpy as np
import os
import os.path as osp
import torch
from torch_geometric.data import Data
from torch_points_kernels.points_cpu import ball_query
import imageio
from tqdm.auto import tqdm

from torch_points3d.core.data_transform import GridSampling3D, SaveOriginalPosId
from torch_geometric.transforms import Compose
import torch_points3d.datasets.registration.fusion as fusion


def to_list(x):
    """
    taken from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/dataset.html#Dataset
    """
    if not isinstance(x, collections.Iterable) or isinstance(x, str):
        x = [x]
    return x


def files_exist(files):
    """
    taken from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/dataset.html#Dataset
    """

    return all([osp.exists(f) for f in files])


def makedirs(path):
    """
    taken from https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/makedirs.py
    """
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def get_urls(filename):
    res = []
    with open(filename, "r") as f:
        res = f.readlines()
    return res


def extract_pcd(depth_image, K, color_image=None):
    """
    transform a depth image into a pointcloud (here a numpy array)
    """

    Z = (depth_image / 1000).ravel()
    mask_z = (Z < 6) * (Z > 0)
    X, Y = np.meshgrid(np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0]))

    Xworld = (X.ravel()[mask_z] + 0.5 - K[0, 2]) * Z[mask_z] / K[0, 0]
    Yworld = (Y.ravel()[mask_z] + 0.5 - K[1, 2]) * Z[mask_z] / K[1, 1]

    pcd = np.vstack((Xworld, Yworld, Z[mask_z])).T
    if color_image is None:
        return pcd
    else:
        color = color_image.reshape(-1, 3)[mask_z, :]
        return pcd, color


def rgbd2pcd(path_img, path_intrinsic, path_trans, path_color=None):

    # read imageio
    depth = imageio.imread(path_img)

    intrinsic = np.loadtxt(path_intrinsic)
    trans = np.loadtxt(path_trans)
    if path_color is not None:
        color_image = imageio.imread(path_color)
        pcd, color = extract_pcd(depth, intrinsic, color_image)
        pcd = pcd.dot(trans[:3, :3].T) + trans[:3, 3]
        return pcd, color
    else:
        pcd = extract_pcd(depth, intrinsic)
        pcd = pcd.dot(trans[:3, :3].T) + trans[:3, 3]
        return pcd


def rgbd2fragment_rough(
    list_path_img,
    path_intrinsic,
    list_path_trans,
    out_path,
    num_frame_per_fragment=5,
    pre_transform=None,
    list_path_color=None,
):

    one_fragment = []
    one_color = []
    ind = 0
    for i, path_img in tqdm(enumerate(list_path_img), total=len(list_path_img)):
        path_trans = list_path_trans[i]
        path_color = None
        if list_path_color is not None:
            path_color = list_path_color[i]
            pcd, color = rgbd2pcd(path_img, path_intrinsic, path_trans, path_color=path_color)
            one_fragment.append(pcd)
            one_color.append(color)
        else:
            pcd = rgbd2pcd(path_img, path_intrinsic, path_trans, path_color=path_color)
            one_fragment.append(pcd)
        if (i + 1) % num_frame_per_fragment == 0:
            pos = torch.from_numpy(np.concatenate(one_fragment, axis=0))
            if list_path_color is None:
                torch_data = Data(pos=pos)
            else:
                color = torch.from_numpy(np.concatenate(one_color, axis=0))
                torch_data = Data(pos=pos, color=color)
            if pre_transform is not None:
                torch_data = pre_transform(torch_data)
            torch.save(torch_data, osp.join(out_path, "fragment_{:06d}.pt".format(ind)))
            ind += 1
            one_fragment = []
            one_color = []
    # concatenate all fragment

    # create batches
    # save fragments for each batches using a simple batch


def filter_pair(pair, dist):
    """
    give a pair of indices where the distance is positive
    """
    pair = pair[dist[:, 0] >= 0]
    if len(pair) > 0:
        pair = pair.numpy()[:, ::-1]
    else:
        pair = np.array([])
    return pair


def compute_overlap_and_matches(data1, data2, max_distance_overlap, reciprocity=False, num_pos=1, trans_gt=torch.eye(4)):

    # we can use ball query on cpu because the points are sorted
    # print(len(data1.pos), len(data2.pos), max_distance_overlap)
    pair, dist = ball_query(
        data2.pos.to(torch.float),
        data1.pos.to(torch.float) @ trans_gt[:3, :3].T + trans_gt[:3, 3],
        radius=max_distance_overlap,
        max_num=num_pos, mode=1, sorted=True)
    pair = filter_pair(pair, dist)
    pair2 = []
    overlap = [pair.shape[0] / len(data1.pos)]
    if reciprocity:
        pair2, dist2 = ball_query(
            data1.pos.to(torch.float) @ trans_gt[:3, :3].T + trans_gt[:3, 3],
            data2.pos.to(torch.float),
            radius=max_distance_overlap,
            max_num=num_pos, mode=1, sorted=True)
        pair2 = filter_pair(pair2, dist2)
        overlap.append(pair2.shape[0] / len(data2.pos))
    # overlap = pair.shape[0] / \
    #    (len(data1.pos) + len(data2.pos) - pair.shape[0])
    # print(pair)

    # print(path1, path2, "overlap=", overlap)
    output = dict(pair=pair, pair2=pair2, overlap=overlap)
    return output

def compute_subsampled_matches(data1, data2, voxel_size=0.1, max_distance_overlap=0.02):
    """
    compute matches on subsampled version of data and track ind
    """
    grid_sampling = Compose([SaveOriginalPosId(), GridSampling3D(voxel_size, mode='last')])
    subsampled_data = grid_sampling(data1.clone())
    origin_id = subsampled_data.origin_id.numpy()
    pair = compute_overlap_and_matches(subsampled_data, data2, max_distance_overlap)['pair']
    pair[:, 0] = origin_id[pair[:, 0]]
    return torch.from_numpy(pair.copy())

def get_3D_bound(list_path_img, path_intrinsic, list_path_trans, depth_thresh, limit_size=600, voxel_size=0.01):
    vol_bnds = np.zeros((3, 2))
    list_min = np.zeros((3, len(list_path_img)))
    list_max = np.zeros((3, len(list_path_img)))
    for i, path_img in tqdm(enumerate(list_path_img), total=len(list_path_img)):
        # read imageio
        depth = imageio.imread(path_img) / 1000.0
        depth[depth > depth_thresh] = 0
        intrinsic = np.loadtxt(path_intrinsic)
        pose = np.loadtxt(list_path_trans[i])
        view_frust_pts = fusion.get_view_frustum(depth, intrinsic, pose)
        list_min[:, i] = np.amin(view_frust_pts, axis=1)
        list_max[:, i] = np.amax(view_frust_pts, axis=1)
    # take the quantile instead of the min to be more robust to outilers frames

    vol_bnds[:, 0] = np.quantile(list_min, 0.1, axis=1)
    vol_bnds[:, 1] = np.quantile(list_max, 0.9, axis=1)

    # remove some voxel that are on the edge to control the size of the tsdf.
    vol_dim = (vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size
    for i in range(3):
        # add and substract delta to limit the size
        if(vol_dim[i] > limit_size):
            delta = voxel_size * (vol_dim[i] - limit_size) * 0.5
            vol_bnds[i][0] += delta
            vol_bnds[i][1] -= delta
    return vol_bnds


def rgbd2fragment_fine(
    list_path_img,
    path_intrinsic,
    list_path_trans,
    out_path,
    num_frame_per_fragment=5,
    voxel_size=0.01,
    pre_transform=None,
    depth_thresh=6,
    save_pc=True,
    limit_size=600
):
    """
    fuse rgbd frame with a tsdf volume and get the mesh using marching cube.
    """

    ind = 0
    begin = 0
    end = num_frame_per_fragment

    vol_bnds = get_3D_bound(list_path_img[begin:end], path_intrinsic, list_path_trans[begin:end], depth_thresh, voxel_size=voxel_size, limit_size=limit_size)

    print(vol_bnds)
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size)
    for i, path_img in tqdm(enumerate(list_path_img), total=len(list_path_img)):

        depth = imageio.imread(path_img).astype(float) / 1000.0
        depth[depth > depth_thresh] = 0
        depth[depth <= 0] = 0
        intrinsic = np.loadtxt(path_intrinsic)
        pose = np.loadtxt(list_path_trans[i])
        tsdf_vol.integrate(depth, intrinsic, pose, obs_weight=1.0)
        if (i + 1) % num_frame_per_fragment == 0:

            if save_pc:
                pcd = tsdf_vol.get_point_cloud(0.35, 0.0)
                torch_data = Data(pos=torch.from_numpy(pcd.copy()))
            else:
                verts, faces, norms = tsdf_vol.get_mesh()
                torch_data = Data(pos=torch.from_numpy(verts.copy()), norm=torch.from_numpy(norms.copy()))
            if pre_transform is not None:
                torch_data = pre_transform(torch_data)
            torch.save(torch_data, osp.join(out_path, "fragment_{:06d}.pt".format(ind)))
            ind += 1

            if i + 1 < len(list_path_img):
                begin = i + 1
                if begin + num_frame_per_fragment < len(list_path_img):
                    end = begin + num_frame_per_fragment
                    vol_bnds = get_3D_bound(
                        list_path_img[begin:end], path_intrinsic, list_path_trans[begin:end], depth_thresh, voxel_size=voxel_size, limit_size=limit_size
                    )
                else:
                    vol_bnds = get_3D_bound(
                        list_path_img[begin:], path_intrinsic, list_path_trans[begin:],
                        depth_thresh, voxel_size=voxel_size, limit_size=limit_size
                    )
                tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size)


class PatchExtractor:
    r"""
    Extract patches on a point cloud
    """

    def __init__(self, radius_patch):
        self.radius_patch = radius_patch

    def __call__(self, data: Data, ind):

        pos = data.pos
        point = pos[ind].view(1, 3)
        ind, dist = ball_query(point, pos, radius=self.radius_patch, max_num=-1, mode=1)

        row, col = ind[dist[:, 0] > 0].t()
        patch = Data()
        for key in data.keys:
            if torch.is_tensor(data[key]):
                if torch.all(col < data[key].shape[0]):
                    patch[key] = data[key][col]

        return patch


def tracked_matches(data_s, data_t, pair):
    """
    allow to keep the index that are still present after a sparse input
    Parameters:
    pair : P x 2 indices of the matched points before any transformation
    """

    pair_np = pair.numpy()
    mask_s = np.isin(pair_np[:, 0], data_s.origin_id.numpy())
    mask_t = np.isin(pair_np[:, 1], data_t.origin_id.numpy())
    # print(data_s.origin_id.shape)
    # print(data_s.pos.shape)
    # print(data_s.xyz.shape)
    mask = np.logical_and(mask_s, mask_t)
    filtered_pair = pair_np[mask]

    table_s = dict(zip(data_s.origin_id.numpy(),
                       np.arange(0, len(data_s.pos))))
    table_t = dict(zip(data_t.origin_id.numpy(),
                       np.arange(0, len(data_t.pos))))
    res = torch.tensor([[table_s[p[0]], table_t[p[1]]] for p in filtered_pair]).to(torch.long)
    return res
