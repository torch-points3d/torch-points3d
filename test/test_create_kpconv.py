import os
import sys
import unittest
import torch

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points3d.applications.kpconv import KPConv
from test.mockdatasets import MockDatasetGeometric
from torch_points3d.core.data_transform import GridSampling

seed = 0
torch.manual_seed(seed)
device = "cpu"


class TestAPI(unittest.TestCase):
    def test_api_arguments(self):
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
            weights=False,
            config=None,
        )
        dataset = MockDatasetGeometric(input_nc + 1, transform=GridSampling(0.01), num_points=1024)
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


if __name__ == "__main__":
    unittest.main()
