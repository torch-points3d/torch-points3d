import os.path as osp
from .base_dataset import BaseDataset
from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
from datasets.utils import contains_key

AVAILABLE_NUMBERS = ["10", "40"]

class ModelNetDataset(BaseDataset):
    def  __init__(self, dataset_opt, training_opt):
        super().__init__(dataset_opt, training_opt)

        number = dataset_opt.number   
        if str(number) not in AVAILABLE_NUMBERS: raise Exception("Only ModelNet10 and ModelNet40 are available")    
        name = 'ModelNet{}'.format(number)
        self._data_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
        pre_transform = T.NormalizeScale()
        transform = T.SamplePoints(dataset_opt.num_points) if contains_key(dataset_opt, "num_points") else None

        train_dataset = ModelNet(
            self._data_path,
            name=str(number),
            train=True,
            transform=transform,
            pre_transform=pre_transform)

        test_dataset = ModelNet(
            self._data_path,
            name=str(number),
            train=False,
            transform=transform,
            pre_transform=pre_transform)

        self.create_dataloaders(train_dataset, test_dataset, validation=None)
    
    def test_dataloader(self):
        return self._test_loader

    def train_dataloader(self):
        return self._train_loader

    @property
    def num_classes(self):
        return self._num_classes
