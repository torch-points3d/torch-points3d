import numpy as np
import torch
import imageio


def extract_pcd(depth, intrinsic):
    return None


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
                  num_frame_per_fragment=50):

    list_fragment = []
    one_fragment = []
    for i, path_img in enumerate(list_path_img):
        path_trans = list_path_trans[i]
        pcd = rgbd2pcd(path_img, path_intrinsic, path_trans)
        one_fragment.append(pcd)
        if i + 1 % num_frame_per_fragment == 0:
            list_fragment.append(np.concatenate(one_fragment, axis=0))
            one_fragment = []
    # concatenate all fragment

    # create batches
    # save fragments for each batches using a simple batch
