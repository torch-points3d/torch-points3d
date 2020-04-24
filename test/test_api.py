import os
import sys
import unittest
import torch

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from test.mockdatasets import MockDatasetGeometric, MockDataset
from torch_points3d.core.data_transform import GridSampling

seed = 0
torch.manual_seed(seed)
device = "cpu"


class TestAPI(unittest.TestCase):
    def test_kpconv(self):
        from torch_points3d.applications.kpconv import KPConv

        input_nc = 3
        num_layers = 4
        grid_sampling = 0.02
        model = KPConv(
            architecture="unet",
            input_nc=input_nc,
            output_nc=5,
            in_feat=32,
            in_grid_size=grid_sampling,
            num_layers=num_layers,
            config=None,
        )
        dataset = MockDatasetGeometric(input_nc + 1, transform=GridSampling(0.01), num_points=128)
        model.set_input(dataset[0], device)
        self.assertEqual(len(model._modules["down_modules"]), num_layers + 1)
        self.assertEqual(len(model._modules["inner_modules"]), 1)
        self.assertEqual(len(model._modules["up_modules"]), 4)

        try:
            model.forward()
        except Exception as e:
            print("Model failing:")
            print(model)
            raise e

    def test_pn2(self):
        from torch_points3d.applications.pointnet2 import PointNet2

        input_nc = 2
        num_layers = 3
        model = PointNet2(
            architecture="unet", input_nc=input_nc, output_nc=5, num_layers=num_layers, multiscale=True, config=None,
        )
        dataset = MockDataset(input_nc, num_points=512)
        model.set_input(dataset[0], device)
        self.assertEqual(len(model._modules["down_modules"]), num_layers - 1)
        self.assertEqual(len(model._modules["inner_modules"]), 1)
        self.assertEqual(len(model._modules["up_modules"]), num_layers)

        try:
            model.forward()
        except Exception as e:
            print("Model failing:")
            print(model)
            raise e

    def test_rsconv(self):
        from torch_points3d.applications.rsconv import RSConv

        input_nc = 2
        num_layers = 4
        model = RSConv(
            architecture="unet", input_nc=input_nc, output_nc=5, num_layers=num_layers, multiscale=True, config=None,
        )
        dataset = MockDataset(input_nc, num_points=1024)
        model.set_input(dataset[0], device)
        self.assertEqual(len(model._modules["down_modules"]), num_layers)
        self.assertEqual(len(model._modules["inner_modules"]), 2)
        self.assertEqual(len(model._modules["up_modules"]), num_layers + 1)

        try:
            model.forward()
        except Exception as e:
            print("Model failing:")
            print(model)
            raise e


if __name__ == "__main__":
    unittest.main()
