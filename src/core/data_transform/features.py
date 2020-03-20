import torch
import torch.nn.functional as F
from torch_geometric.transforms import Center


def compute_planarity(eigenvalues):
    """
    compute the planarity with respect to the eigenvalues of the covariance matrix of the centered pointcloud
    let $\lambda_1, \lambda_2, \lambda_3$ be the eigenvalues st
    $$
    \lambda_1 \leq \lambda_2 \leq \lambda_3
    $$
    then planarity is defined as:
    planarity = \frac{\lambda_2 - \lambda_1}{\lambda_3}
    """
    return (eigenvalues[1] - eigenvalues[0]) / eigenvalues[2]


class NormalFeature(object):
    """
    add normal as feature. if it doesn't exist, compute normals
    using PCA
    """
    def __call__(self, data):
        if data.norm is None:
            raise NotImplementedError("TODO: Implement normal computation")

        norm = data.norm
        if data.x is None:
            data.x = norm
        else:
            data.x = torch.cat([data.x, norm], -1)
        return data


class PCACompute(object):
    """
    compute PCA of a point cloud and store the eigen values and the eigenvectors
    """

    def call(self, data):
        pos_centered = data.pos - data.pos.mean(axis=0)
        cov_matrix = pos_centered.T.mm(pos_centered) / len(pos_centered)
        eig, v = torch.symeig(cov_matrix, eigenvectors=True)
        data.eigenvalues = eig
        data.eigenvectors = v
        return data


class PlanarityFilter(object):
    """
    compute planarity and return false if the planarity is above a threshold
    """

    def __init__(self, thresh=0.3):
        self.thresh = thresh

    def __call__(self, data):
        if(data.eigenvalues is None):
            data = PCACompute()(data)
        planarity = compute_planarity(data.eigenvalues)
        return planarity < self.thresh
