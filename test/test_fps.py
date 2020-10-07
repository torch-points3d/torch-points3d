import torch
import numpy as np
from torch_geometric.nn import fps
import unittest
import logging
import os
import sys

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, ".."))

from test import run_if_cuda
from torch_points3d.datasets.registration.utils import fps_sampling

log = logging.getLogger(__name__)


class TestPytorchClusterFPS(unittest.TestCase):
    @run_if_cuda
    def test_simple(self):
        num_points = 2048
        pos = torch.randn((num_points, 3)).cuda()
        batch = torch.zeros((num_points)).cuda().long()
        idx = fps(pos, batch, 0.25)

        idx = idx.detach().cpu().numpy()

        cnd_1 = np.sum(idx) > 0
        cnd_2 = np.sum(idx) < num_points * idx.shape[0]

        assert (
            cnd_1 and cnd_2
        ), "Your Pytorch Cluster FPS doesn't seem to return the correct value. It shouldn't be used to perform sampling"

    def test_fps_sampling_registration(self):
        pos = torch.tensor([[0, 0, 0], [0.5, 0.5, 0], [0.4, 0.2, 0], [2, 2, 2], [-1, -2, -0.01]]).float()
        pair_ind = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]).long()
        num_pos_pairs = 3
        new_ind = fps_sampling(pair_ind, pos, num_pos_pairs)
        new_pair_ind = pair_ind[new_ind]
        sol = torch.tensor([[0, 0], [3, 3], [4, 4]]).long()
        torch.testing.assert_allclose(new_pair_ind, sol)


if __name__ == "__main__":
    unittest.main()
