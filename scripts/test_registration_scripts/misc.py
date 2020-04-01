import numpy as np


def read_gt_log(path):
    """
    read the gt.log of evaluation set of 3DMatch or ETH Dataset and parse it.
    """
    list_pair = []
    list_mat = []
    with open(path, "r") as f:
        all_mat = f.readlines()
    mat = np.zeros((4, 4))
    for i in range(len(all_mat)):
        if i % 5 == 0:
            if i != 0:
                list_mat.append(mat)
            mat = np.zeros((4, 4))
            list_pair.append(list(map(int, all_mat[i].split("\t")[:-1])))
        else:
            line = all_mat[i].split("\t")

            mat[i % 5 - 1] = np.asarray(line[:4], dtype=np.float)
    list_mat.append(mat)
    return list_pair, list_mat
