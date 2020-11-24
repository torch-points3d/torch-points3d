import torch
import random


def euler_angles_to_rotation_matrix(theta, random_order=False):
    R_x = torch.tensor(
        [[1, 0, 0], [0, torch.cos(theta[0]), -torch.sin(theta[0])], [0, torch.sin(theta[0]), torch.cos(theta[0])]]
    )

    R_y = torch.tensor(
        [[torch.cos(theta[1]), 0, torch.sin(theta[1])], [0, 1, 0], [-torch.sin(theta[1]), 0, torch.cos(theta[1])]]
    )

    R_z = torch.tensor(
        [[torch.cos(theta[2]), -torch.sin(theta[2]), 0], [torch.sin(theta[2]), torch.cos(theta[2]), 0], [0, 0, 1]]
    )

    matrices = [R_x, R_y, R_z]
    if random_order:
        random.shuffle(matrices)
    R = torch.mm(matrices[2], torch.mm(matrices[1], matrices[0]))
    return R


def get_cross_product_matrix(k):
    return torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], device=k.device)


def rodrigues(axis, theta):
    """
    given an axis of norm one and an angle, compute the rotation matrix using rodrigues formula
    source : https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    K = get_cross_product_matrix(axis)
    t = torch.tensor([theta], device=axis.device)
    R = torch.eye(3, device=axis.device) + torch.sin(t) * K + (1 - torch.cos(t)) * K.mm(K)
    return R


def get_trans(x):
    """
    get the rotation matrix from the vector representation using the rodrigues formula
    """
    T = torch.eye(4, device=x.device)
    T[:3, 3] = x[3:]
    axis = x[:3]
    theta = torch.norm(axis)
    if theta > 0:
        axis = axis / theta
    T[:3, :3] = rodrigues(axis, theta)
    return T
