"""
registration toolbox (algorithm for some registration algorithm)
Implemented: fast_global_registration
teaser
"""

import torch
from torch_points3d.utils.geometry import get_trans
from torch_geometric.nn import knn


def get_matches(feat_source, feat_target, sym=False):

    matches = knn(feat_target, feat_source, k=1).T
    if sym:
        match_inv = knn(feat_source, feat_target, k=1).T
        mask = match_inv[matches[:, 1], 1] == torch.arange(matches.shape[0], device=feat_source.device)
        return matches[mask]
    else:
        return matches


def estimate_transfo(xyz, xyz_target):
    """
    estimate the rotation and translation using Kabsch algorithm
    Parameters:
    xyz :
    xyz_target:
    """
    assert xyz.shape == xyz.shape
    xyz_c = xyz - xyz.mean(0)
    xyz_target_c = xyz_target - xyz_target.mean(0)
    Q = xyz_c.T.mm(xyz_target_c) / len(xyz)
    U, S, V = torch.svd(Q)
    d = torch.det(V.mm(U.T))
    diag = torch.diag(torch.tensor([1, 1, d], device=xyz.device))
    R = V.mm(diag).mm(U.T)
    t = xyz_target.mean(0) - R @ xyz.mean(0)
    T = torch.eye(4, device=xyz.device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def get_geman_mclure_weight(xyz, xyz_target, mu):
    """
    compute the weights defined here for the iterative reweighted least square.
    http://vladlen.info/papers/fast-global-registration.pdf
    """
    norm2 = torch.norm(xyz_target - xyz, dim=1) ** 2
    return (mu / (mu + norm2)).view(-1, 1)


def get_matrix_system(xyz, xyz_target, weight):
    """
    Build matrix of size 3N x 6 and b of size 3N

    xyz size N x 3
    xyz_target size N x 3
    weight size N
    the matrix is minus cross product matrix concatenate with the identity (rearanged).
    """
    assert xyz.shape == xyz_target.shape
    A_x = torch.zeros(xyz.shape[0], 6, device=xyz.device)
    A_y = torch.zeros(xyz.shape[0], 6, device=xyz.device)
    A_z = torch.zeros(xyz.shape[0], 6, device=xyz.device)
    b_x = weight.view(-1) * (xyz_target[:, 0] - xyz[:, 0])
    b_y = weight.view(-1) * (xyz_target[:, 1] - xyz[:, 1])
    b_z = weight.view(-1) * (xyz_target[:, 2] - xyz[:, 2])
    A_x[:, 1] = weight.view(-1) * xyz[:, 2]
    A_x[:, 2] = -weight.view(-1) * xyz[:, 1]
    A_x[:, 3] = weight.view(-1) * 1
    A_y[:, 0] = -weight.view(-1) * xyz[:, 2]
    A_y[:, 2] = weight.view(-1) * xyz[:, 0]
    A_y[:, 4] = weight.view(-1) * 1
    A_z[:, 0] = weight.view(-1) * xyz[:, 1]
    A_z[:, 1] = -weight.view(-1) * xyz[:, 0]
    A_z[:, 5] = weight.view(-1) * 1
    return torch.cat([A_x, A_y, A_z], 0), torch.cat([b_x, b_y, b_z], 0).view(-1, 1)


def fast_global_registration(xyz, xyz_target, mu_init=1, num_iter=20):
    """
    estimate the rotation and translation using Fast Global Registration algorithm (M estimator for robust estimation)
    http://vladlen.info/papers/fast-global-registration.pdf
    """
    assert xyz.shape == xyz_target.shape

    T_res = torch.eye(4, device=xyz.device)
    mu = mu_init
    source = xyz.clone()
    weight = torch.ones(len(source), 1, device=xyz.device)
    for i in range(num_iter):
        if i > 0 and i % 5 == 0:
            mu /= 2.0
        A, b = get_matrix_system(source, xyz_target, weight)
        x, _ = torch.solve(A.T @ b, A.T.mm(A))
        T = get_trans(x.view(-1))
        source = source.mm(T[:3, :3].T) + T[:3, 3]
        T_res = T @ T_res
        weight = get_geman_mclure_weight(source, xyz_target, mu)
    return T_res


def teaser_pp_registration(
    xyz,
    xyz_target,
    noise_bound=0.05,
    cbar2=1,
    rotation_gnc_factor=1.4,
    rotation_max_iterations=100,
    rotation_cost_threshold=1e-12,
):
    assert xyz.shape == xyz_target.shape
    import teaserpp_python

    # Populating the parameters
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = cbar2
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = rotation_gnc_factor
    solver_params.rotation_max_iterations = rotation_max_iterations
    solver_params.rotation_cost_threshold = rotation_cost_threshold

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    solver.solve(xyz.T.detach().cpu().numpy(), xyz_target.T.detach().cpu().numpy())

    solution = solver.getSolution()
    T_res = torch.eye(4, device=xyz.device)
    T_res[:3, :3] = torch.from_numpy(solution.rotation).to(xyz.device)
    T_res[:3, 3] = torch.from_numpy(solution.translation).to(xyz.device)
    return T_res
