import unittest
from omegaconf import OmegaConf
import os
import sys
from glob import glob
import torch
from torch_geometric.data.data import Data

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

from test.mockdatasets import MockDataset, MockDatasetConfig
from test.mock_models import MockModel, MockModelConfig
from src.datasets.base_dataset import BaseDataset
from src.models.model_factory import instantiate_model
from src.utils.model_building_utils.model_definition_resolver import resolve_model
from src.utils.enums import ConvolutionFormat


class MockBaseDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        self._data_path = dataset_opt.dataroot
        self.train_dataset = MockDataset()
        self.val_dataset = MockDataset()


class TestDataLoaders(unittest.TestCase):
    def test_list(self):
        dataset_opt = MockDatasetConfig()
        setattr(dataset_opt, "dataroot", os.path.join(DIR, "temp_dataset"))

        mock_base_dataset = MockBaseDataset(dataset_opt)
        mock_base_dataset.test_dataset = [MockDataset() for i in range(5)]
        model_config = MockModelConfig()
        setattr(model_config, "conv_type", "dense")
        model = MockModel(model_config)

        mock_base_dataset.create_dataloaders(model, 2, True, 0, False)
        datasets = mock_base_dataset.test_dataloaders()

        self.assertEqual(len(datasets), 5)

    def test_normal(self):
        dataset_opt = MockDatasetConfig()
        setattr(dataset_opt, "dataroot", os.path.join(DIR, "temp_dataset"))

        mock_base_dataset = MockBaseDataset(dataset_opt)
        mock_base_dataset.test_dataset = MockDataset()
        model_config = MockModelConfig()
        setattr(model_config, "conv_type", "dense")
        model = MockModel(model_config)

        mock_base_dataset.create_dataloaders(model, 2, True, 0, False)
        datasets = mock_base_dataset.test_dataloaders()

        self.assertEqual(len(datasets), 1)


class TestBatchCollate(unittest.TestCase):
    def test_num_batches(self):
        data = Data(pos=torch.randn((2, 3, 3)))
        self.assertEqual(MockBaseDataset.get_num_samples(data, ConvolutionFormat.DENSE.value[-1]), 2)

        data = Data(pos=torch.randn((3, 3)), batch=torch.tensor([0, 1, 2]))
        self.assertEqual(MockBaseDataset.get_num_samples(data, ConvolutionFormat.PARTIAL_DENSE.value[-1]), 3)

    def test_get_sample(self):
        data = Data(pos=torch.randn((2, 3, 3)))
        torch.testing.assert_allclose(
            MockBaseDataset.get_sample(data, "pos", 1, ConvolutionFormat.DENSE.value[-1]), data.pos[1]
        )

        data = Data(pos=torch.randn((3, 3)), batch=torch.tensor([0, 1, 2]))
        torch.testing.assert_allclose(
            MockBaseDataset.get_sample(data, "pos", 1, ConvolutionFormat.PARTIAL_DENSE.value[-1]), data.pos[1]
        )


if __name__ == "__main__":
    unittest.main()
