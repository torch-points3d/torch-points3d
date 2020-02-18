import torch


def fitting_loss(sq_distance, radius):
    """ KPConv fitting loss. For each query point it ensures that at least one neighboor is
    close to each kernel point

    Arguments:
        sq_distance - For each querry point, from all neighboors to all KP points [N_querry, N_neighboors, N_KPoints]
        radius - Radius of the convolution
    """
    kpmin = sq_distance.min(dim=1)[0]
    normalised_kpmin = kpmin / (radius ** 2)
    return torch.mean(normalised_kpmin)


def repulsion_loss(deformed_kpoints, radius):
    """ Ensures that the deformed points within the kernel remain equidistant

    Arguments:
        deformed_kpoints - deformed points for each query point
        radius - Radius of the kernel
    """
    deformed_kpoints / float(radius)
    n_points = deformed_kpoints.shape[1]
    repulsive_loss = 0
    for i in range(n_points):
        with torch.no_grad():
            other_points = torch.cat([deformed_kpoints[:, :i, :], deformed_kpoints[:, i + 1 :, :]], dim=1)
        distances = torch.sqrt(torch.sum((other_points - deformed_kpoints[:, i : i + 1, :]) ** 2, dim=-1))
        repulsion_force = torch.sum(torch.pow(torch.relu(1.5 - distances), 2), dim=1)
        repulsive_loss += torch.mean(repulsion_force)
    return repulsive_loss


def permissive_loss(deformed_kpoints, radius):
    """This loss is responsible to penalize deformed_kpoints to
    move outside from the radius defined for the convolution
    """
    norm_deformed_normalized = torch.norm(deformed_kpoints, p=2, dim=-1) / float(radius)
    permissive_loss = torch.mean(norm_deformed_normalized[norm_deformed_normalized > 1.0])
    return permissive_loss
