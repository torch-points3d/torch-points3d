import os
import unittest
import numpy as np
import numpy.testing as npt
import torch
from torch_geometric.data import Data

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

from torch_points3d.core.data_transform.transforms import MeshToNormal


class TestModelUtils(unittest.TestCase):
    def setUp(self):

        pos = np.array(
            [[0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0]]
        )  # Should be [0, 0, 1], [1, 0, 0]
        face = np.array([[0, 1, 2], [3, 4, 5]]).T

        self.data = Data(pos=torch.from_numpy(pos).float(), face=torch.from_numpy(face))

    def test_mesh_to_normal(self):
        mesh_transform = MeshToNormal()
        data = mesh_transform(self.data)
        normals = data.normals.numpy()
        npt.assert_array_equal(normals[0], [0, 0, 1])
        npt.assert_array_equal(normals[1], [-1, 0, 0])
