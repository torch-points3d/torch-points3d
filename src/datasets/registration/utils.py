import numpy as np
import os.path as osp
import torch
from torch_geometric.data import Data
from torch_points.points_cpu import ball_query
import imageio
from tqdm import tqdm


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


def rgbd2fragment(list_path_img,
                  path_intrinsic, list_path_trans,
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


def compute_overlap_and_matches(list_fragment_path, out_dir,
                                max_distance_overlap=0.1):
    ind = 0
    for path1 in list_fragment_path:
        data1 = torch.load(path1)
        for path2 in list_fragment_path:
            if path1 < path2:
                out_path = osp.join(out_dir, 'matches{:06d}.npy'.format(ind))
                ind += 1
                data2 = torch.load(path2)

                # we can use ball query on cpu because the point are sorted
                pair, dist = ball_query(data1.pos, data2.pos,
                                        radius=max_distance_overlap,
                                        max_num=1, mode=1)

                if(len(pair) > 0):
                    pair = pair.detach().numpy()[:, ::-1]
                else:
                    pair = np.array([])
                overlap = pair.shape[0] / \
                    (len(data1.pos) + len(data2.pos) - pair.shape[0])
                output = dict(pair=pair,
                              path_source=path1,
                              path_target=path2,
                              overlap=overlap)
                np.save(out_path, output)
