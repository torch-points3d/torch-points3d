from abc import ABC, abstractmethod
import logging

# A logger for this file
log = logging.getLogger(__name__)

class BaseDataset(ABC):
    def __init__(self, opt):
        self._opt = opt

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

