from abc import ABC, abstractmethod
import logging
from functools import partial

import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.transforms import Compose, FixedPoints
from torch_geometric.data import Batch, DataLoader, Dataset

from datasets.transforms import MultiScaleTransform
from datasets.batch import SimpleBatch


# A logger for this file
log = logging.getLogger(__name__)


class BaseDataset:
    def __init__(self, dataset_opt, training_opt):
        self.dataset_opt = dataset_opt
        self.training_opt = training_opt
        self.strategies = {}
        self._torch_loader = training_opt.use_torch_loader

    def _create_dataloaders(self, train_dataset, test_dataset, validation=None):
        """ Creates the data loaders. Must be called in order to complete the setup of the Dataset
        """
        self._num_classes = train_dataset.num_classes
        self._feature_dimension = train_dataset.num_features
        if self._torch_loader:
            dataloader = partial(
                torch.utils.data.DataLoader,
                pin_memory=True,
                collate_fn=lambda data_list: SimpleBatch.from_data_list(data_list),
            )
        else:
            dataloader = DataLoader
        self._train_loader = dataloader(
            train_dataset,
            batch_size=self.training_opt.batch_size,
            shuffle=self.training_opt.shuffle,
            num_workers=self.training_opt.num_workers,
        )

        self._test_loader = dataloader(
            test_dataset,
            batch_size=self.training_opt.batch_size,
            shuffle=False,
            num_workers=self.training_opt.num_workers,
        )

    def test_dataloader(self):
        return self._test_loader

    @property
    def has_fixed_points_transform(self):
        """
        This property checks if the dataset contains T.FixedPoints transform, meaning the number of points is fixed
        """
        transform_train = self._train_loader.dataset.transform
        transform_test = self._test_loader.dataset.transform

        if transform_train is None or transform_test is None:
            return False

        if not isinstance(transform_train, Compose):
            transform_train = Compose([transform_train])

        if not isinstance(transform_test, Compose):
            transform_test = Compose([transform_test])

        train_bool = False
        test_bool = False

        for transform in transform_train.transforms:
            if isinstance(transform, FixedPoints):
                train_bool = True
        for transform in transform_test.transforms:
            if isinstance(transform, FixedPoints):
                test_bool = True
        return train_bool and test_bool

    def train_dataloader(self):
        return self._train_loader

    @property
    def is_hierarchical(self):
        """ Used by the metric trackers to log hierarchical metrics
        """
        return False

    @property
    def class_to_segments(self):
        """ Use this property to return the hierarchical map between classes and segment ids, example:
        {
            'Airplaine': [0,1,2],
            'Boat': [3,4,5]
        } 
        """
        return None

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def weight_classes(self):
        return getattr(self._train_loader.dataset, "weight_classes", None)

    @property
    def feature_dimension(self):
        return self._feature_dimension

    @property
    def batch_size(self):
        return self.training_opt.batch_size

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
                        setattr(
                            attr.dataset, "transform", T.Compose([current_transform, transform]),
                        )

    def set_strategies(self, model, precompute_multi_scale=False):
        strategies = model.get_sampling_and_search_strategies()
        transform = MultiScaleTransform(strategies, precompute_multi_scale)
        self._set_multiscale_transform(transform)
