from torch_points3d.datasets.object_detection.scannet import ScannetDataset
from torch_points3d.core.data_transform import GridSampling3D
from torch_points3d.applications.pretrained_api import PretainedRegistry
from test.mockdatasets import MockDatasetGeometric, MockDataset
import os
import sys
import unittest
import torch
from omegaconf import OmegaConf

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


seed = 0
torch.manual_seed(seed)
device = "cpu"


class TestAPIUnet(unittest.TestCase):
    def test_kpconv(self):
        from torch_points3d.applications.kpconv import KPConv

        input_nc = 3
        num_layers = 4
        grid_sampling = 0.02
        in_feat = 32
        model = KPConv(
            architecture="unet",
            input_nc=input_nc,
            in_feat=in_feat,
            in_grid_size=grid_sampling,
            num_layers=num_layers,
            config=None,
        )
        dataset = MockDatasetGeometric(input_nc + 1, transform=GridSampling3D(0.01), num_points=128)
        self.assertEqual(len(model._modules["down_modules"]), num_layers + 1)
        self.assertEqual(len(model._modules["inner_modules"]), 1)
        self.assertEqual(len(model._modules["up_modules"]), 4)
        self.assertFalse(model.has_mlp_head)
        self.assertEqual(model.output_nc, in_feat)

        try:
            data_out = model.forward(dataset[0])
            self.assertEqual(data_out.x.shape[1], in_feat)
        except Exception as e:
            print("Model failing:")
            print(model)
            raise e

        input_nc = 3
        num_layers = 4
        grid_sampling = 0.02
        in_feat = 32
        output_nc = 5
        model = KPConv(
            architecture="unet",
            input_nc=input_nc,
            output_nc=output_nc,
            in_feat=in_feat,
            in_grid_size=grid_sampling,
            num_layers=num_layers,
            config=None,
        )
        dataset = MockDatasetGeometric(input_nc + 1, transform=GridSampling3D(0.01), num_points=128)
        self.assertEqual(len(model._modules["down_modules"]), num_layers + 1)
        self.assertEqual(len(model._modules["inner_modules"]), 1)
        self.assertEqual(len(model._modules["up_modules"]), 4)
        self.assertTrue(model.has_mlp_head)
        self.assertEqual(model.output_nc, output_nc)

        try:
            data_out = model.forward(dataset[0])
            self.assertEqual(data_out.x.shape[1], output_nc)
        except Exception as e:
            print("Model failing:")
            print(model)
            raise e

    def test_pn2(self):
        from torch_points3d.applications.pointnet2 import PointNet2

        input_nc = 2
        num_layers = 3
        output_nc = 5
        model = PointNet2(
            architecture="unet",
            input_nc=input_nc,
            output_nc=output_nc,
            num_layers=num_layers,
            multiscale=True,
            config=None,
        )
        dataset = MockDataset(input_nc, num_points=512)
        self.assertEqual(len(model._modules["down_modules"]), num_layers - 1)
        self.assertEqual(len(model._modules["inner_modules"]), 1)
        self.assertEqual(len(model._modules["up_modules"]), num_layers)

        try:
            data_out = model.forward(dataset[0])
            self.assertEqual(data_out.x.shape[1], output_nc)
        except Exception as e:
            print("Model failing:")
            print(model)
            raise e

    def test_rsconv(self):
        from torch_points3d.applications.rsconv import RSConv

        input_nc = 2
        num_layers = 4
        output_nc = 5
        model = RSConv(
            architecture="unet",
            input_nc=input_nc,
            output_nc=output_nc,
            num_layers=num_layers,
            multiscale=True,
            config=None,
        )
        dataset = MockDataset(input_nc, num_points=1024)
        self.assertEqual(len(model._modules["down_modules"]), num_layers)
        self.assertEqual(len(model._modules["inner_modules"]), 2)
        self.assertEqual(len(model._modules["up_modules"]), num_layers)

        try:
            data_out = model.forward(dataset[0])
            self.assertEqual(data_out.x.shape[1], output_nc)
        except Exception as e:
            print("Model failing:")
            print(model)
            raise e

    def test_sparseconv3d(self):
        from torch_points3d.applications.sparseconv3d import SparseConv3d

        input_nc = 3
        num_layers = 4
        in_feat = 32
        out_feat = in_feat * 3
        model = SparseConv3d(
            architecture="unet", input_nc=input_nc, in_feat=in_feat, num_layers=num_layers, config=None,
        )
        dataset = MockDatasetGeometric(input_nc, transform=GridSampling3D(0.01, quantize_coords=True), num_points=128)
        self.assertEqual(len(model._modules["down_modules"]), num_layers + 1)
        self.assertEqual(len(model._modules["inner_modules"]), 1)
        self.assertEqual(len(model._modules["up_modules"]), 4 + 1)
        self.assertFalse(model.has_mlp_head)
        self.assertEqual(model.output_nc, out_feat)

        try:
            data_out = model.forward(dataset[0])
            self.assertEqual(data_out.x.shape[1], out_feat)
        except Exception as e:
            print("Model failing:")
            print(model)
            print(e)

        input_nc = 3
        num_layers = 4

        output_nc = 5
        model = SparseConv3d(
            architecture="unet", input_nc=input_nc, output_nc=output_nc, num_layers=num_layers, config=None,
        )
        dataset = MockDatasetGeometric(input_nc, transform=GridSampling3D(0.01, quantize_coords=True), num_points=128)
        self.assertEqual(len(model._modules["down_modules"]), num_layers + 1)
        self.assertEqual(len(model._modules["inner_modules"]), 1)
        self.assertEqual(len(model._modules["up_modules"]), 4 + 1)
        self.assertTrue(model.has_mlp_head)
        self.assertEqual(model.output_nc, output_nc)

        try:
            data_out = model.forward(dataset[0])
            self.assertEqual(data_out.x.shape[1], output_nc)
        except Exception as e:
            print("Model failing:")
            print(model)
            raise e


class TestAPIEncoder(unittest.TestCase):
    def test_kpconv(self):
        from torch_points3d.applications.kpconv import KPConv

        input_nc = 3
        num_layers = 4
        grid_sampling = 0.02
        in_feat = 16
        model = KPConv(
            architecture="encoder",
            input_nc=input_nc,
            in_feat=in_feat,
            in_grid_size=grid_sampling,
            num_layers=num_layers,
            config=None,
        )
        dataset = MockDatasetGeometric(input_nc + 1, transform=GridSampling3D(0.01), num_points=128)
        self.assertEqual(len(model._modules["down_modules"]), num_layers + 1)
        self.assertEqual(len(model._modules["inner_modules"]), 1)
        self.assertFalse(model.has_mlp_head)
        self.assertEqual(model.output_nc, 32 * in_feat)

        try:
            data_out = model.forward(dataset[0])
            self.assertEqual(data_out.x.shape[1], 32 * in_feat)
        except Exception as e:
            print("Model failing:")
            print(model)
            raise e

        input_nc = 3
        num_layers = 4
        grid_sampling = 0.02
        in_feat = 32
        output_nc = 5
        model = KPConv(
            architecture="encoder",
            input_nc=input_nc,
            output_nc=output_nc,
            in_feat=in_feat,
            in_grid_size=grid_sampling,
            num_layers=num_layers,
            config=None,
        )
        dataset = MockDatasetGeometric(input_nc + 1, transform=GridSampling3D(0.01), num_points=128)
        self.assertEqual(len(model._modules["down_modules"]), num_layers + 1)
        self.assertEqual(len(model._modules["inner_modules"]), 1)
        self.assertTrue(model.has_mlp_head)
        self.assertEqual(model.output_nc, output_nc)

        try:
            data_out = model.forward(dataset[0])
            self.assertEqual(data_out.x.shape[1], output_nc)
        except Exception as e:
            print("Model failing:")
            print(model)
            raise e

    def test_pn2(self):
        from torch_points3d.applications.pointnet2 import PointNet2

        input_nc = 2
        num_layers = 3
        output_nc = 5
        model = PointNet2(
            architecture="encoder",
            input_nc=input_nc,
            output_nc=output_nc,
            num_layers=num_layers,
            multiscale=True,
            config=None,
        )
        dataset = MockDataset(input_nc, num_points=512)
        self.assertEqual(len(model._modules["down_modules"]), num_layers - 1)
        self.assertEqual(len(model._modules["inner_modules"]), 1)

        try:
            data_out = model.forward(dataset[0])
            self.assertEqual(data_out.x.shape[1], output_nc)
        except Exception as e:
            print("Model failing:")
            print(model)
            raise e

    def test_rsconv(self):
        from torch_points3d.applications.rsconv import RSConv

        input_nc = 2
        num_layers = 4
        output_nc = 5
        model = RSConv(
            architecture="encoder",
            input_nc=input_nc,
            output_nc=output_nc,
            num_layers=num_layers,
            multiscale=True,
            config=None,
        )
        dataset = MockDataset(input_nc, num_points=1024)
        self.assertEqual(len(model._modules["down_modules"]), num_layers)
        self.assertEqual(len(model._modules["inner_modules"]), 1)

        try:
            data_out = model.forward(dataset[0])
            self.assertEqual(data_out.x.shape[1], output_nc)
        except Exception as e:
            print("Model failing:")
            print(model)
            raise e

    def test_minkowski(self):
        from torch_points3d.applications.minkowski import Minkowski

        input_nc = 3
        num_layers = 4
        in_feat = 16
        model = Minkowski(
            architecture="encoder", input_nc=input_nc, in_feat=in_feat, num_layers=num_layers, config=None,
        )
        dataset = MockDatasetGeometric(input_nc, transform=GridSampling3D(0.01, quantize_coords=True), num_points=128)
        self.assertEqual(len(model._modules["down_modules"]), num_layers + 1)
        self.assertEqual(len(model._modules["inner_modules"]), 1)
        self.assertFalse(model.has_mlp_head)
        self.assertEqual(model.output_nc, 8 * in_feat)

        try:
            data_out = model.forward(dataset[0])
            # self.assertEqual(data_out.x.shape[1], 8 * in_feat)
        except Exception as e:
            print("Model failing:")
            print(model)
            raise e

        input_nc = 3
        num_layers = 4
        grid_sampling = 0.02
        in_feat = 32
        output_nc = 5
        model = Minkowski(
            architecture="encoder",
            input_nc=input_nc,
            output_nc=output_nc,
            in_feat=in_feat,
            in_grid_size=grid_sampling,
            num_layers=num_layers,
            config=None,
        )
        dataset = MockDatasetGeometric(input_nc, transform=GridSampling3D(0.01, quantize_coords=True), num_points=128)
        self.assertEqual(len(model._modules["down_modules"]), num_layers + 1)
        self.assertEqual(len(model._modules["inner_modules"]), 1)
        self.assertTrue(model.has_mlp_head)
        self.assertEqual(model.output_nc, output_nc)

        try:
            data_out = model.forward(dataset[0])
            self.assertEqual(data_out.x.shape[1], output_nc)
        except Exception as e:
            print("Model failing:")
            print(model)
            raise e


if __name__ == "__main__":
    unittest.main()
