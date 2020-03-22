import torch
import numpy as np
from torch_geometric.nn import fps
import unittest
import logging

from . import run_if_cuda

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


if __name__ == "__main__":
    unittest.main()
