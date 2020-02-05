import numpy as np
import os.path as osp
import torch
from torch_geometric.data import Data
from torch_points.points_cpu import ball_query
import imageio
from tqdm import tqdm

import src.datasets.registration.fusion as fusion


def extract_pcd(depth_image, K, color_image=None):
    """
    transform a depth image into a pointcloud (here a numpy array)
    """

    Z = (depth_image/1000).ravel()
    mask_z = (Z < 6) * (Z > 0)
    X, Y = np.meshgrid(np.arange(depth_image.shape[1]),
                       np.arange(depth_image.shape[0]))

    Xworld = (X.ravel()[mask_z] + 0.5 - K[0, 2])*Z[mask_z]/K[0, 0]
    Yworld = (Y.ravel()[mask_z] + 0.5 - K[1, 2])*Z[mask_z]/K[1, 1]

    pcd = np.vstack((Xworld, Yworld, Z[mask_z])).T
    if(color_image is None):
        return pcd
    else:
        color = color_image.reshape(-1, 3)[mask_z, :]
        return pcd, color


def rgbd2pcd(path_img, path_intrinsic, path_trans, path_color=None):

    # read imageio
    depth = imageio.imread(path_img)

    intrinsic = np.loadtxt(path_intrinsic)
    trans = np.loadtxt(path_trans)
    if(path_color is not None):
        color_image = imageio.imread(path_color)
        pcd, color = extract_pcd(depth, intrinsic, color_image)
        pcd = pcd.dot(trans[:3, :3].T) + trans[:3, 3]
        return pcd, color
    else:
        pcd = extract_pcd(depth, intrinsic)
        pcd = pcd.dot(trans[:3, :3].T) + trans[:3, 3]
        return pcd


def rgbd2fragment_rough(list_path_img,
                        path_intrinsic,
                        list_path_trans,
                        out_path,
                        num_frame_per_fragment=5,
                        pre_transform=None,
                        list_path_color=None):

    one_fragment = []
    one_color = []
    ind = 0
    for i, path_img in tqdm(enumerate(list_path_img), total=len(list_path_img)):
        path_trans = list_path_trans[i]
        path_color = None
        if(list_path_color is not None):
            path_color = list_path_color[i]
            pcd, color = rgbd2pcd(path_img, path_intrinsic,
                                  path_trans, path_color=path_color)
            one_fragment.append(pcd)
            one_color.append(color)
        else:
            pcd = rgbd2pcd(path_img, path_intrinsic,
                           path_trans, path_color=path_color)
            one_fragment.append(pcd)
        if (i + 1) % num_frame_per_fragment == 0:
            pos = torch.from_numpy(np.concatenate(one_fragment, axis=0))
            if(list_path_color is None):
                torch_data = Data(pos=pos)
            else:
                color = torch.from_numpy(np.concatenate(one_color, axis=0))
                torch_data = Data(pos=pos, color=color)
            if pre_transform is not None:
                torch_data = pre_transform(torch_data)
            torch.save(torch_data, osp.join(out_path,
                                            'fragment_{:06d}.pt'.format(ind)))
            ind += 1
            one_fragment = []
            one_color = []
    # concatenate all fragment

    # create batches
    # save fragments for each batches using a simple batch


def compute_overlap_and_matches(path1, path2,
                                max_distance_overlap):
    data1 = torch.load(path1)
    data2 = torch.load(path2)


    # we can use ball query on cpu because the points are sorted
    # print(len(data1.pos), len(data2.pos), max_distance_overlap)
    pair, dist = ball_query(data1.pos, data2.pos,
                            radius=max_distance_overlap,
                            max_num=1, mode=1)
    pair = pair[dist[:, 0] > 0]
    if(len(pair) > 0):
        pair = pair.numpy()[:, ::-1]
    else:
        pair = np.array([])
    #overlap = pair.shape[0] / \
    #    (len(data1.pos) + len(data2.pos) - pair.shape[0])
    # print(pair)
    overlap = [pair.shape[0]/len(data1.pos), pair.shape[0]/len(data2.pos)]
    # print(path1, path2, "overlap=", overlap)
    output = dict(pair=pair,
                  path_source=path1,
                  path_target=path2,
                  overlap=overlap)
    return output


def get_3D_bound(list_path_img,
                 path_intrinsic, list_path_trans):
    vol_bnds = np.zeros((3,2))
    for i, path_img in tqdm(enumerate(list_path_img), total=len(list_path_img)):
        # read imageio
        depth = imageio.imread(path_img) / 1000.0
        depth[depth > 6] = 0
        intrinsic = np.loadtxt(path_intrinsic)
        pose = np.loadtxt(list_path_trans[i])
        view_frust_pts = fusion.get_view_frustum(depth, intrinsic, pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    return vol_bnds


def rgbd2fragment_fine(list_path_img,
                       path_intrinsic,
                       list_path_trans,
                       list_path_color,
                       out_path,
                       num_frame_per_fragment=5,
                       voxel_size=0.01,
                       pre_transform=None):
    """
    fuse rgbd frame with a tsdf volume and get the mesh using marching cube.
    """

    ind = 0
    begin = 0
    end = num_frame_per_fragment
    vol_bnds = get_3D_bound(list_path_img[begin:end],
                            path_intrinsic,
                            list_path_trans[begin:end])

    print(vol_bnds)
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size)
    for i, path_img in tqdm(enumerate(list_path_img), total=len(list_path_img)):

        depth = imageio.imread(path_img).astype(float) / 1000.0
        depth[depth > 6] = 0
        depth[depth <= 0] = 0
        intrinsic = np.loadtxt(path_intrinsic)
        color_image = imageio.imread(list_path_color[i])
        pose = np.loadtxt(list_path_trans[i])
        tsdf_vol.integrate(color_image, depth,
                           intrinsic, pose,
                           obs_weight=1.)
        if((i+1) % num_frame_per_fragment == 0):
            verts, faces, norms, colors = tsdf_vol.get_mesh()
            torch_data = Data(pos=torch.from_numpy(verts.copy()),
                              color=torch.from_numpy(colors),
                              norm=torch.from_numpy(norms.copy()))
            if pre_transform is not None:
                torch_data = pre_transform(torch_data)
            torch.save(torch_data, osp.join(out_path,
                                            'fragment_{:06d}.pt'.format(ind)))
            ind += 1

            if(i+1 < len(list_path_img)):
                begin = i+1
                if(begin + num_frame_per_fragment < len(list_path_img)):
                    end = begin + num_frame_per_fragment
                    vol_bnds = get_3D_bound(list_path_img[begin:end],
                                            path_intrinsic,
                                            list_path_trans[begin:end])
                else:
                    vol_bnds = get_3D_bound(list_path_img[begin:],
                                            path_intrinsic,
                                            list_path_trans[begin:])
                tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size)







class PatchExtractor:
    r"""
    Extract patches on a point cloud
    """

    def __init__(self, radius_patch):
        self.radius = radius_patch

    def __call__(self, data: Data, ind):

        pos = data.pos
        point = pos[ind].view(1, 3)
        ind, dist = ball_query(point, pos,
                               radius=self.radius_patch,
                               max_num=-1, mode=1)

        row, col = ind.t()

        for key in data.keys:
            if(torch.is_tensor(data[key])):
                data[key] = data[key][row]
        return data
