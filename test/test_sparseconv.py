import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

import torch_points3d.modules.SparseConv3d as sp3d
import torch_points3d.modules.SparseConv3d.modules


class TestSparseConv(unittest.TestCase):
    def test_setbackend(self):
        sp3d.nn.set_backend("torchsparse")
        c = sp3d.nn.Conv3d(1, 1)
        sp3d.nn.set_backend("minkowski")
        self.assertNotIsInstance(sp3d.nn.Conv3d(1, 1), type(c))
        c = sp3d.nn.Conv3d(1, 1)
        sp3d.nn.set_backend("torchsparse")
        self.assertNotIsInstance(sp3d.nn.Conv3d(1, 1), type(c))

    def test_weights(self):
        sp3d.nn.set_backend("torchsparse")
        s = sp3d.modules.ResBlock(3, 10, sp3d.nn.Conv3d)
        sp3d.nn.set_backend("minkowski")
        m = sp3d.modules.ResBlock(3, 10, sp3d.nn.Conv3d)
        self.assertNotIsInstance(m.block[0], type(s.block[0]))
        m.load_state_dict(s.state_dict())


if __name__ == "__main__":
    unittest.main()
