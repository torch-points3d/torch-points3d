import unittest
from omegaconf import DictConfig, OmegaConf
import os
import sys
from glob import glob
import torch
import numpy as np
from torch_geometric.data.data import Data
import torch_geometric.transforms as T

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

from test.mockdatasets import MockDataset, MockDatasetConfig
from test.mock_models import MockModel, MockModelConfig
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.models.model_factory import instantiate_model
from torch_points3d.utils.model_building_utils.model_definition_resolver import resolve_model
from torch_points3d.utils.enums import ConvolutionFormat
import torch_points3d.core.data_transform as T3d


class Options:
    def __init__(self):
        pass

    def get(self, key, default):
        if hasattr(self, key):
            return getattr(self, key, default)
        else:
            return default

    def keys(self):
        return self.__dict__.keys()


class CustomMockDataset:
    def __init__(self, num_points, input_nc, output_nc, num_samples, conv_type="dense"):
        self.num_points = num_points
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.num_samples = num_samples
        self.conv_type = conv_type

    def __len__(self):
        return self.num_samples

    @property
    def num_classes(self):
        return self.output_nc

    @property
    def num_features(self):
        return self.input_nc

    def __getitem__(self, idx):
        pos = torch.from_numpy(np.random.normal(0, 1, (self.num_points, 3)))
        y = torch.from_numpy(np.random.normal(0, 1, (self.num_points, self.output_nc)))
        x = torch.from_numpy(np.random.normal(0, 1, (self.num_points, self.input_nc)))
        return Data(x=x, pos=pos, y=y)


class MockBaseDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        self._data_path = dataset_opt.dataroot
        self.train_dataset = MockDataset()
        self.val_dataset = MockDataset()


class TestDataset(unittest.TestCase):
    def test_empty_dataset(self):
        opt = Options()
        opt.dataset_name = os.path.join(os.getcwd(), "test")
        opt.dataroot = os.path.join(os.getcwd(), "test")
        opt.pre_transform = [DictConfig({"transform": "RandomNoise"})]
        opt.test_transform = [DictConfig({"transform": "AddOnes"})]
        opt.val_transform = [DictConfig({"transform": "Jitter"})]
        opt.train_transform = [DictConfig({"transform": "RandomSymmetry"})]
        dataset = BaseDataset(opt)

        self.assertEqual(str(dataset.pre_transform), str(T.Compose([T3d.RandomNoise()])))
        self.assertEqual(str(dataset.test_transform), str(T.Compose([T3d.AddOnes()])))
        self.assertEqual(str(dataset.train_transform), str(T.Compose([T3d.RandomSymmetry()])))
        self.assertEqual(str(dataset.val_transform), str(T.Compose([T3d.Jitter()])))
        self.assertEqual(str(dataset.inference_transform), str(T.Compose([T3d.RandomNoise(), T3d.AddOnes()])))
        self.assertEqual(dataset.train_dataset, None)
        self.assertEqual(dataset.test_dataset, None)
        self.assertEqual(dataset.val_dataset, None)

    def test_simple_datasets(self):
        opt = Options()
        opt.dataset_name = os.path.join(os.getcwd(), "test")
        opt.dataroot = os.path.join(os.getcwd(), "test")

        class SimpleDataset(BaseDataset):
            def __init__(self, dataset_opt):
                super(SimpleDataset, self).__init__(dataset_opt)

                self.train_dataset = CustomMockDataset(10, 1, 3, 10)
                self.test_dataset = CustomMockDataset(10, 1, 3, 10)

        dataset = SimpleDataset(opt)

        model_config = MockModelConfig()
        model_config.conv_type = "dense"
        model = MockModel(model_config)
        dataset.create_dataloaders(model, 5, True, 0, False)

        self.assertEqual(dataset.pre_transform, None)
        self.assertEqual(dataset.test_transform, None)
        self.assertEqual(dataset.train_transform, None)
        self.assertEqual(dataset.val_transform, None)
        self.assertNotEqual(dataset.train_dataset, None)
        self.assertNotEqual(dataset.test_dataset, None)
        self.assertTrue(dataset.has_test_loaders)
        self.assertFalse(dataset.has_val_loader)

    def test_multiple_test_datasets(self):
        opt = Options()
        opt.dataset_name = os.path.join(os.getcwd(), "test")
        opt.dataroot = os.path.join(os.getcwd(), "test")

        class MultiTestDataset(BaseDataset):
            def __init__(self, dataset_opt):
                super(MultiTestDataset, self).__init__(dataset_opt)

                self.train_dataset = CustomMockDataset(10, 1, 3, 10)
                self.val_dataset = CustomMockDataset(10, 1, 3, 10)
                self.test_dataset = [CustomMockDataset(10, 1, 3, 10), CustomMockDataset(10, 1, 3, 20)]

        dataset = MultiTestDataset(opt)

        model_config = MockModelConfig()
        model_config.conv_type = "dense"
        model = MockModel(model_config)
        dataset.create_dataloaders(model, 5, True, 0, False)

        loaders = dataset.test_dataloaders
        self.assertEqual(len(loaders), 2)
        self.assertEqual(len(loaders[0].dataset), 10)
        self.assertEqual(len(loaders[1].dataset), 20)
        self.assertEqual(dataset.num_classes, 3)
        self.assertEqual(dataset.is_hierarchical, False)
        self.assertEqual(dataset.has_val_loader, True)
        self.assertEqual(dataset.class_to_segments, None)
        self.assertEqual(dataset.feature_dimension, 1)

        batch = next(iter(loaders[0]))
        num_samples = BaseDataset.get_num_samples(batch, "dense")
        self.assertEqual(num_samples, 5)

        sample = BaseDataset.get_sample(batch, "pos", 1, "dense")
        self.assertEqual(sample.shape, (10, 3))
        sample = BaseDataset.get_sample(batch, "x", 1, "dense")
        self.assertEqual(sample.shape, (10, 1))
        self.assertEqual(dataset.num_batches, {"train": 2, "val": 2, "test_0": 2, "test_1": 4})

    def test_normal(self):
        dataset_opt = MockDatasetConfig()
        setattr(dataset_opt, "dataroot", os.path.join(DIR, "temp_dataset"))

        mock_base_dataset = MockBaseDataset(dataset_opt)
        mock_base_dataset.test_dataset = MockDataset()
        model_config = MockModelConfig()
        setattr(model_config, "conv_type", "dense")
        model = MockModel(model_config)

        mock_base_dataset.create_dataloaders(model, 2, True, 0, False)
        datasets = mock_base_dataset.test_dataloaders

        self.assertEqual(len(datasets), 1)

    def test_get_by_name(self):
        dataset_opt = MockDatasetConfig()
        setattr(dataset_opt, "dataroot", os.path.join(DIR, "temp_dataset"))

        mock_base_dataset = MockBaseDataset(dataset_opt)
        mock_base_dataset.test_dataset = [MockDataset(), MockDataset()]
        mock_base_dataset.train_dataset = MockDataset()
        mock_base_dataset.val_dataset = MockDataset()

        for name in ["train", "val", "test_0", "test_1"]:
            self.assertEqual(mock_base_dataset.get_dataset(name).name, name)

        test_with_name = MockDataset()
        setattr(test_with_name, "name", "testos")
        mock_base_dataset.test_dataset = test_with_name
        with self.assertRaises(ValueError):
            mock_base_dataset.get_dataset("test_1")
        mock_base_dataset.get_dataset("testos")

        with self.assertRaises(ValueError):
            mock_base_dataset.test_dataset = [test_with_name, test_with_name]

    def test_add_weights(self):
        dataset_opt = MockDatasetConfig()
        setattr(dataset_opt, "dataroot", os.path.join(DIR, "temp_dataset"))

        mock_base_dataset = MockBaseDataset(dataset_opt)
        mock_base_dataset.train_dataset.data = Data(y=torch.tensor([1, 1, 1, 0]))
        mock_base_dataset.add_weights()
        self.assertGreater(mock_base_dataset.weight_classes[0], mock_base_dataset.weight_classes[1])

        mock_base_dataset.add_weights(class_weight_method="log")
        print(mock_base_dataset.weight_classes)
        self.assertGreater(mock_base_dataset.weight_classes[0], mock_base_dataset.weight_classes[1])


class TestBatchCollate(unittest.TestCase):
    def test_num_batches(self):
        data = Data(pos=torch.randn((2, 3, 3)))
        self.assertEqual(MockBaseDataset.get_num_samples(data, ConvolutionFormat.DENSE.value), 2)

        data = Data(pos=torch.randn((3, 3)), batch=torch.tensor([0, 1, 2]))
        self.assertEqual(MockBaseDataset.get_num_samples(data, ConvolutionFormat.PARTIAL_DENSE.value), 3)

    def test_get_sample(self):
        data = Data(pos=torch.randn((2, 3, 3)))
        torch.testing.assert_allclose(
            MockBaseDataset.get_sample(data, "pos", 1, ConvolutionFormat.DENSE.value), data.pos[1]
        )

        data = Data(pos=torch.randn((3, 3)), batch=torch.tensor([0, 1, 2]))
        torch.testing.assert_allclose(
            MockBaseDataset.get_sample(data, "pos", 1, ConvolutionFormat.PARTIAL_DENSE.value), data.pos[1]
        )


if __name__ == "__main__":
    unittest.main()
