import os
import sys
import unittest
import numpy as np

import torch
from torch_geometric.data import Data

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ROOT)

from src.datasets.base_dataset import BaseDataset
from mock_models import MockModelConfig, MockModel

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

class CustomMockDataset():

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

class TestBaseDataset(unittest.TestCase):
    def test_empty_dataset(self):

        opt = Options()
        opt.dataset_name = os.path.join(os.getcwd(), "test")
        opt.dataroot = os.path.join(os.getcwd(), "test")
        
        dataset = BaseDataset(opt)

        self.assertEqual(dataset.pre_transform, None)
        self.assertEqual(dataset.test_transform, None)
        self.assertEqual(dataset.train_transform, None)
        self.assertEqual(dataset.val_transform, None)
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
                self.val_dataset = CustomMockDataset(10, 1, 3, 10)
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
        self.assertNotEqual(dataset.val_dataset, None)

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

        loaders = dataset.test_dataloaders()
        self.assertEqual(len(loaders), 2)
        self.assertEqual(len(loaders[0].dataset), 10)
        self.assertEqual(len(loaders[1].dataset), 20)
        self.assertEqual(dataset.num_classes, 3)
        self.assertEqual(dataset.is_hierarchical, False)
        self.assertEqual(dataset.has_fixed_points_transform, False)
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
        self.assertEqual(dataset.num_batches, {'train': 2, 'val': 2, 'test_0': 2, 'test_1': 4})

        repr = 'Dataset: MultiTestDataset \n\x1b[0;95mpre_transform \x1b[0m= None\n\x1b[0;95mtest_transform \x1b[0m= None\n\x1b[0;95mtrain_transform \x1b[0m= None\n\x1b[0;95mval_transform \x1b[0m= None\n\x1b[0;95minference_transform \x1b[0m= None\nSize of \x1b[0;95mtrain_dataset \x1b[0m= 10\nSize of \x1b[0;95mtest_dataset \x1b[0m= 10, 20\nSize of \x1b[0;95mval_dataset \x1b[0m= 10\n\x1b[0;95mBatch size =\x1b[0m 5'
        self.assertEqual(dataset.__repr__(), repr)

if __name__ == "__main__":
    unittest.main()
