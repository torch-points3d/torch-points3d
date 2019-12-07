from abc import ABC, abstractmethod
import logging
from torch_geometric.data import DataLoader

# A logger for this file
log = logging.getLogger(__name__)

class BaseDataset():
    def __init__(self, dataset_opt, training_opt):
        self.dataset_opt = dataset_opt
        self.training_opt = training_opt

    def create_dataloaders(self, train_dataset,  test_dataset, validation=None):
        self._num_classes = train_dataset.num_classes
        self._train_loader = DataLoader(train_dataset, batch_size=4, shuffle=self.training_opt.shuffle,
                                num_workers=self.training_opt.num_workers)

        self._test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False,
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

