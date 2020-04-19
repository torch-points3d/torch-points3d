import unittest
import torch
from torch_geometric.data import Data

import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)

from torch_points3d.datasets.batch import SimpleBatch


class TestSimpleBatch(unittest.TestCase):
    def test_fromlist(self):
        nb_points = 100
        pos = torch.randn((nb_points, 3))
        y = torch.tensor([range(10) for i in range(pos.shape[0])], dtype=torch.float)
        d = Data(pos=pos, y=y)

        b = SimpleBatch.from_data_list([d, d])
        self.assertEqual(b.pos.size(), (2, 100, 3))
        self.assertEqual(b.y.size(), (2, 100, 10))


if __name__ == "__main__":
    unittest.main()
