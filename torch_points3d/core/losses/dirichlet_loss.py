import torch
import torch_points_kernels as tp
from torch_cluster import radius
from torch_scatter import scatter_add

_MAX_NEIGHBOURS = 32


class DirichletLoss(torch.nn.Module):
    """ L2 norm of the gradient estimated as the average change of a field value f
    accross neighbouring points within a radius r
    """

    def __init__(self, r, aggr=torch.mean):
        super().__init__()
        self._r = r
        self._aggr = aggr

    def forward(self, pos, f, batch_idx=None):
        """ Computes the Dirichlet loss (or L2 norm of the gradient) of f
        Arguments:
            pos -- [N,3] (or [B,N,3] for dense format)  location of each point
            f -- [N] (or [B,N] for dense format)  Value of a function at each points
            batch_idx -- [N] Batch id of each point (Only for sparse format)
        """
        return dirichlet_loss(self._r, pos, f, batch_idx=batch_idx, aggr=self._aggr)


def dirichlet_loss(r, pos, f, batch_idx=None, aggr=torch.mean):
    """ Computes the Dirichlet loss (or L2 norm of the gradient) of f
    Arguments:
        r -- Radius for the beighbour search
        pos -- [N,3] (or [B,N,3] for dense format)  location of each point
        f -- [N] (or [B,N] for dense format)  Value of a function at each points
        batch_idx -- [N] Batch id of each point (Only for sparse format)
        aggr -- aggregation function for the final loss value
    """
    if batch_idx is None:
        assert f.dim() == 2 and pos.dim() == 3
        return _dirichlet_dense(r, pos, f, aggr)
    else:
        assert f.dim() == 1 and pos.dim() == 2
        return _dirichlet_sparse(r, pos, f, batch_idx, aggr)


def _dirichlet_dense(r, pos, f, aggr):
    variances = _variance_estimator_dense(r, pos, f)
    return 1 / 2.0 * aggr(variances)


def _variance_estimator_dense(r, pos, f):
    nei_idx = tp.ball_query(r, _MAX_NEIGHBOURS, pos, pos, sort=True)[0].reshape(pos.shape[0], -1).long()  # [B,N * nei]
    f_neighboors = f.gather(1, nei_idx).reshape(f.shape[0], f.shape[1], -1)  # [B,N , nei]
    gradient = (f.unsqueeze(-1).repeat(1, 1, f_neighboors.shape[-1]) - f_neighboors) ** 2  # [B,N,nei]
    return gradient.sum(-1)


def _dirichlet_sparse(r, pos, f, batch_idx, aggr):
    variances = _variance_estimator_sparse(r, pos, f, batch_idx)
    return 1 / 2.0 * aggr(variances)


def _variance_estimator_sparse(r, pos, f, batch_idx):
    with torch.no_grad():
        assign_index = radius(pos, pos, r, batch_x=batch_idx, batch_y=batch_idx)
        y_idx, x_idx = assign_index
        # diff = pos[x_idx] - pos[y_idx]
        # squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
        # weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

        grad_f = (f[x_idx] - f[y_idx]) ** 2
    y = scatter_add(grad_f, y_idx, dim=0, dim_size=pos.size(0))
    return y
