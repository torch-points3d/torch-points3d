import os
from .base_dataset import BaseDataset
from torch_geometric.datasets import S3DIS
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T


class S3DISDataset(BaseDataset):
    def  __init__(self, dataset_opt, training_opt):
        super().__init__(dataset_opt, training_opt)
        self._data_path = os.path.join(dataset_opt.dataroot, 'S3DIS')
        train_dataset = S3DIS(self._data_path, test_area=self.dataset_opt.fold, train=True,
                                pre_transform=None)
        test_dataset = S3DIS( self._data_path, test_area=self.dataset_opt.fold, train=False,
                                pre_transform=None)

        self.create_dataloaders(train_dataset, test_dataset, validation=None)
    
    def test_dataloader(self):
        return self._test_loader

    def train_dataloader(self):
        return self._train_loader

    @property
    def num_classes(self):
        return self._num_classes