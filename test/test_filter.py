import unittest
import sys
import os
import numpy as np
import torch
from torch_geometric.data import Data

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, ".."))

from src.core.data_transform import FCompose, PlanarityFilter, euler_angles_to_rotation_matrix


class TestFilter(unittest.TestCase):
    def test_planarity_filter(self):

        # Plane with high planarity
        vec1 = torch.randn(3)
        vec1 = vec1 / torch.norm(vec1)
        vec2 = torch.randn(3)
        vec2 = vec2 / torch.norm(vec2)
        plane = torch.randn(100, 1) * vec1.view(1, 3) + torch.randn(100, 1) * vec2.view(1, 3)
        data1 = Data(pos=plane)
        # random isotropic gaussian
        data2 = Data(pos=torch.randn(100, 3))
        plane_filter = PlanarityFilter(0.3)
        self.assertTrue(plane_filter(data2))
        self.assertFalse(plane_filter(data1))

    def test_composition(self):

        U_1 = euler_angles_to_rotation_matrix(torch.rand(3) * np.pi)
        U_2 = euler_angles_to_rotation_matrix(torch.rand(3) * np.pi)
        U_3 = euler_angles_to_rotation_matrix(torch.rand(3) * np.pi)
        p_1 = torch.rand(1000, 3) @ U_1.T @ torch.diag(torch.tensor([1, 0.7, 0.5])) @ U_1
        p_2 = torch.rand(1000, 3) @ U_2.T @ torch.diag(torch.tensor([1, 0.9, 0.001])) @ U_2
        p_3 = torch.rand(1000, 3) @ U_3.T @ torch.diag(torch.tensor([1, 0.0001, 0.000001])) @ U_3

        data_1 = Data(pos=p_1)
        data_2 = Data(pos=p_2)
        data_3 = Data(pos=p_3)

        compose_filter = FCompose([PlanarityFilter(0.5, is_leq=True), PlanarityFilter(0.1, is_leq=False)])

        self.assertTrue(compose_filter(data_1))
        self.assertFalse(compose_filter(data_2))
        self.assertFalse(compose_filter(data_3))
