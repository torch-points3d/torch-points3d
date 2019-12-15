import os
from .base_dataset import BaseDataset
from torch_geometric.datasets import ShapeNet
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T


class ShapeNetDataset(BaseDataset):
    def  __init__(self, dataset_opt, training_opt):
        super().__init__(dataset_opt, training_opt)
        self._data_path = os.path.join(dataset_opt.dataroot, 'ShapeNet')
        self._category = dataset_opt.category
        pre_transform = T.NormalizeScale()
        train_dataset = ShapeNet(self._data_path, self._category, train=True,
                                pre_transform=pre_transform)
        test_dataset = ShapeNet( self._data_path, self._category, train=False,
                                pre_transform=pre_transform)

        self.create_dataloaders(train_dataset, test_dataset, validation=None)
    
    def test_dataloader(self):
        return self._test_loader

    def train_dataloader(self):
        return self._train_loader

    @property
    def num_classes(self):
        return self._num_classes
