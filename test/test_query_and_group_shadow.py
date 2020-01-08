import os
import sys
import unittest
import numpy as np
import numpy.testing as npt
import torch
from torch_geometric.data import Data

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(ROOT)

from models.pointnet2_customkernel.modules import QueryAndGroupShadow


class TestQueryAndGroupShadow(unittest.TestCase):
    r"""
    Parameters
    ----------
    xyz : torch.Tensor
        xyz coordinates of the features (B, N, 3)
    new_xyz : torch.Tensor
        centriods (B, npoint, 3)
    features : torch.Tensor
        Descriptors of the features (B, C, N)
    Returns
    -------
    new_features : torch.Tensor
        (B, 3 + C, npoint, nsample) tensor
    """

    def test_simple(self):

        self._feat_dim = 4

        self._xyz = torch.from_numpy(np.asarray([[[0., 0., 0.], [1., 0., 0.], [1., 0., 1.]]])).float().cuda()
        self._new_xyz = torch.from_numpy(np.asarray([[[0., 0., 0.]]])).float().cuda()
        self._features = torch.randn((1, self._feat_dim, 3)).float().cuda()

        shadow_point = -1 * torch.ones((3, ))
        query_and_group_shadow = QueryAndGroupShadow(0.1, 2, shadow_point=shadow_point, set_zero=False)

        out = query_and_group_shadow(self._xyz, self._new_xyz, self._features)

        answer = torch.cat([shadow_point, torch.zeros((self._feat_dim, ))]).cpu().numpy()

        npt.assert_array_equal(out[0, :, :, -1].squeeze().detach().cpu().numpy(), answer)

    def test_simple_2(self):

        self._batch_size = 2
        self._feat_dim = 4
        self._xyz = torch.from_numpy(np.asarray([[[0., 0., 0.], [1., 0., 0.], [1., 0., 1.]], [
                                     [0.2, 0.2, 0.5], [1., -1., -2.], [0., 0., 0.]]])).float().cuda()
        self._new_xyz = torch.from_numpy(np.asarray([[[0.1, 0.1, 0.1]], [[0.1, 0.2, 0.1]]])).float().cuda()
        self._features = torch.randn((self._batch_size, self._feat_dim, 3)).float().cuda()

        shadow_point = -1 * torch.ones((3, ))
        query_and_group_shadow = QueryAndGroupShadow(0.1, 2, shadow_point=shadow_point, set_zero=True)

        out = query_and_group_shadow(self._xyz, self._new_xyz, self._features)

        out_npy = out.detach().cpu().numpy()
        npt.assert_array_equal(out_npy, np.zeros_like(out_npy))


if __name__ == '__main__':
    unittest.main()
