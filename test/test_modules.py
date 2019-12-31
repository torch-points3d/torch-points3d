import unittest

import sys 
import os 

import torch
import numpy.testing as npt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(ROOT)

from models.PointNet.modules import AffineSTNkD

class TestPointnet(unittest.TestCase):

    # test that stn forward works and is initialised with the identity
    def test_stn(self):
        pos = torch.tensor([
            [1, 1, 2],
            [-1, 0, 1],
            [10, 12, 13],
            [-18, 15, 16]
        ]).to(torch.float32)
        batch = torch.tensor([0, 0, 1, 1])

        stn = AffineSTNkD(num_batches=2)

        trans_pos = stn(pos, batch)

        npt.assert_array_equal(np.asarray(pos.detach()), np.asarray(trans_pos.detach()))        


if __name__ == "__main__":
    unittest.main()