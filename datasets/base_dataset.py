from abc import ABC, abstractmethod
import logging
from torch_geometric.data import DataLoader
from .transforms import MultiScaleTransform

# A logger for this file
log = logging.getLogger(__name__)

class BaseDataset():
    def __init__(self, dataset_opt, training_opt):
        self.dataset_opt = dataset_opt
        self.training_opt = training_opt
        self.strategies = {}

    def create_dataloaders(self, train_dataset,  test_dataset, validation=None):
        self._num_classes = train_dataset.num_classes
        self._train_loader = DataLoader(train_dataset, batch_size=self.training_opt.batch_size, shuffle=self.training_opt.shuffle,
                                num_workers=self.training_opt.num_workers)

        self._test_loader = DataLoader(test_dataset, batch_size=self.training_opt.batch_size, shuffle=False,
                                num_workers=self.training_opt.num_workers)
    @abstractmethod
    def test_dataloader(self):
        pass

    @abstractmethod
    def train_dataloader(self):
        pass

    @property
    @abstractmethod
    def num_classes(self):
        pass

    def set_transform(self):
        for attr_name, attr in self.__dict__.items():
            if "loader" in attr_name:
                transform = attr.dataset.transform
                if transform is None:
                    attr.dataset.transform = self.transform
                else:
                    raise NotImplementedError('Merging of transform not implemented yet')

    def set_strategies(self, strategies, precompute_multi_scale=False):
        self.transform = MultiScaleTransform(strategies, precompute_multi_scale)


    
