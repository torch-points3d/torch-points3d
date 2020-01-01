from abc import ABC, abstractmethod
import logging

import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Batch, DataLoader, Dataset

from datasets.transforms import MultiScaleTransform


# A logger for this file
log = logging.getLogger(__name__)


class BaseDataset():
    def __init__(self, dataset_opt, training_opt):
        self.dataset_opt = dataset_opt
        self.training_opt = training_opt
        self.strategies = {}

    def _create_dataloaders(self, train_dataset,  test_dataset, validation=None):
        """ Creates the data loaders. Must be called in order to complete the setup of the Dataset
        """
        self._num_classes = train_dataset.num_classes
        self._feature_dimension = train_dataset.num_features
        self._train_loader = DataLoader(train_dataset, batch_size=self.training_opt.batch_size, shuffle=self.training_opt.shuffle,
                                        num_workers=self.training_opt.num_workers)

        self._test_loader = DataLoader(test_dataset, batch_size=self.training_opt.batch_size, shuffle=False,
                                       num_workers=self.training_opt.num_workers)

    def test_dataloader(self):
        return self._test_loader

    def train_dataloader(self):
        return self._train_loader

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def weight_classes(self):
        return getattr(self._train_loader.dataset, "weight_classes", None)

    @property
    def feature_dimension(self):
        return self._feature_dimension

    def _set_multiscale_transform(self, transform):
        for _, attr in self.__dict__.items():
            if isinstance(attr, DataLoader):
                current_transform = getattr(attr.dataset, "transform", None)
                if current_transform is None:
                    setattr(attr.dataset, "transform", transform)
                else:
                    if isinstance(current_transform, T.Compose):  # The transform contains several transformations
                        current_transform.transforms += [transform]
                    else:
                        setattr(attr.dataset, "transform", T.Compose([current_transform, transform]))

    def set_strategies(self, model, precompute_multi_scale=False):
        strategies = model.get_sampling_and_search_strategies()
        transform = MultiScaleTransform(strategies, precompute_multi_scale)
        self._set_multiscale_transform(transform)
