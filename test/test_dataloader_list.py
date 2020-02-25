import unittest
from omegaconf import OmegaConf
import os
import sys
from glob import glob
import torch

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

from test.mockdatasets import MockDataset, MockDatasetConfig
from test.mock_models import MockModel, MockModelConfig
from src.datasets.base_dataset import BaseDataset
from src.models.model_factory import instantiate_model
from src.utils.model_building_utils.model_definition_resolver import resolve_model

class TestBaseDataset(unittest.TestCase):
    def test_list(self):

        dataset_opt = MockDatasetConfig()
        setattr(dataset_opt, "dataroot", os.path.join(DIR, "temp_dataset"))

        class MockBaseDataset(BaseDataset):
            
            def __init__(self, dataset_opt):
                super().__init__(dataset_opt)

                self.train_dataset = MockDataset()

                self.val_dataset = MockDataset()

                self.test_dataset = [MockDataset() for i in range(5)]

        mock_base_dataset = MockBaseDataset(dataset_opt)

        model_config = MockModelConfig()
        setattr(model_config, "conv_type", "dense")
        model = MockModel(model_config)

        mock_base_dataset.create_dataloaders(model, 2, True, 0, False)
        datasets = mock_base_dataset.test_dataloaders()

        self.assertEqual(len(datasets), 5)

    def test_normal(self):

        dataset_opt = MockDatasetConfig()
        setattr(dataset_opt, "dataroot", os.path.join(DIR, "temp_dataset"))

        class MockBaseDataset(BaseDataset):
            
            def __init__(self, dataset_opt):
                super().__init__(dataset_opt)

                self.train_dataset = MockDataset()

                self.val_dataset = MockDataset()

                self.test_dataset = MockDataset()

        mock_base_dataset = MockBaseDataset(dataset_opt)

        model_config = MockModelConfig()
        setattr(model_config, "conv_type", "dense")
        model = MockModel(model_config)

        mock_base_dataset.create_dataloaders(model, 2, True, 0, False)
        datasets = mock_base_dataset.test_dataloaders()

        self.assertEqual(len(datasets), 1)

if __name__ == "__main__":
    unittest.main()
