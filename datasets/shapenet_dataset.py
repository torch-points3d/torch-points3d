import os
from .base_dataset import BaseDataset
from torch_geometric.datasets import ShapeNet
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T


class ShapeNetDataset(BaseDataset):
    def  __init__(self, opt):
        super().__init__(opt)
        self._data_path = os.path.join(opt.dataroot, 'ShapeNet')
        self._category = opt.shapenet.category
        transform = T.Compose([
            T.RandomTranslate(0.01),
            T.RandomRotate(15, axis=0),
            T.RandomRotate(15, axis=1),
            T.RandomRotate(15, axis=2)
        ])
        pre_transform = T.NormalizeScale()
        train_dataset = ShapeNet(self._data_path, self._category, train=True, transform=transform,
                                pre_transform=pre_transform)
        test_dataset = ShapeNet( self._data_path, self._category, train=False,
                                pre_transform=pre_transform)
        self._num_classes = train_dataset.num_classes
        self._train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                                num_workers=6)

        self._test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False,
                                num_workers=6)

    
    def test_dataloader(self):
        return self._test_loader

    def train_dataloader(self):
        return self._test_loader

    @property
    def num_classes(self):
        return self._num_classes
