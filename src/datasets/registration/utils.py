import numpy as np
import os.path as osp
import torch
import imageio


def extract_pcd(depth_image, K):
    """
    transform a depth image into a pointcloud (here a numpy array)
    """

    Z = (depth_image/1000).ravel()
    mask_z = (Z < 6) * (Z > 0)
    X, Y = np.meshgrid(np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0]))

    Xworld = (X.ravel()[mask_z] + 0.5 - K[0, 2])*Z[mask_z]/K[0, 0]
    Yworld = (Y.ravel()[mask_z] + 0.5 - K[1, 2])*Z[mask_z]/K[1, 1]

    pcd = np.vstack((Xworld, Yworld, Z[mask_z])).T
    return pcd



def rgbd2pcd(path_img, path_intrinsic, path_trans):

    # read imageio
    depth = imageio.imread(path_img)
    intrinsic = np.loadtxt(path_intrinsic)
    trans = np.loadtxt(path_trans)
    pcd = extract_pcd(depth, intrinsic)
    pcd = pcd.dot(trans[:3, :3].T) + trans[:3, 3]
    return pcd


def rgbd2fragment(list_path_img,
                  path_intrinsic, list_path_trans,
                  out_path,
                  num_frame_per_fragment=50):

    one_fragment = []
    ind = 0
    for i, path_img in enumerate(list_path_img):
        path_trans = list_path_trans[i]
        pcd = rgbd2pcd(path_img, path_intrinsic, path_trans)
        one_fragment.append(pcd)
        if i + 1 % num_frame_per_fragment == 0:
            torch_pcd = torch.from_numpy(np.concatenate(one_fragment, axis=0))
            torch.save(torch_pcd, osp.join(out_path,
                                           'fragment_{}'.format(ind)))
            ind += 1
            one_fragment = []
    # concatenate all fragment

    # create batches
    # save fragments for each batches using a simple batch
