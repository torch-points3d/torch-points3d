import os
from abc import ABC, abstractmethod
import logging
from functools import partial

import torch
import torch_geometric
from torch_geometric.transforms import Compose, FixedPoints

from src.core.data_transform import instantiate_transforms, MultiScaleTransform
from src.datasets.batch import SimpleBatch
from src.datasets.multiscale_data import MultiScaleBatch
from src.utils.enums import ConvolutionFormat
from src.utils.colors import COLORS
from src.models.base_model import BaseModel

# A logger for this file
log = logging.getLogger(__name__)


class BaseDataset:
    def __init__(self, dataset_opt):
        self.dataset_opt = dataset_opt

        # Default dataset path
        class_name = self.__class__.__name__.lower().replace("dataset", "")
        self._data_path = os.path.join(dataset_opt.dataroot, class_name)
        self._batch_size = None
        self.strategies = {}

        self.train_sampler = None
        self.test_sampler = None
        self.val_sampler = None

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.pre_transform = None
        self.test_transform = None
        self.train_transform = None
        self.val_transform = None

        for key_name in dataset_opt.keys():
            if "transform" in key_name:
                new_name = key_name.replace("transforms", "transform")
                try:
                    transform = instantiate_transforms(getattr(dataset_opt, key_name))
                except Exception:
                    log.exception("Error trying to create {}".format(new_name))
                    continue
                setattr(self, new_name, transform)

    @staticmethod
    def _get_collate_function(conv_type, is_multiscale):
        if is_multiscale:
            if conv_type.lower() == ConvolutionFormat.PARTIAL_DENSE.value[-1].lower():
                return lambda datalist: MultiScaleBatch.from_data_list(datalist)
            else:
                raise NotImplementedError()

        if (
            conv_type.lower() == ConvolutionFormat.PARTIAL_DENSE.value[-1].lower()
            or conv_type.lower() == ConvolutionFormat.MESSAGE_PASSING.value[-1].lower()
        ):
            return lambda datalist: torch_geometric.data.batch.Batch.from_data_list(datalist)
        elif conv_type.lower() == ConvolutionFormat.DENSE.value[-1].lower():
            return lambda datalist: SimpleBatch.from_data_list(datalist)

    def create_dataloaders(
        self, model: BaseModel, batch_size: int, shuffle: bool, num_workers: int, precompute_multi_scale: bool,
    ):
        """ Creates the data loaders. Must be called in order to complete the setup of the Dataset
        """
        conv_type = model.conv_type
        self._batch_size = batch_size
        batch_collate_function = BaseDataset._get_collate_function(conv_type, precompute_multi_scale)
        dataloader = partial(torch.utils.data.DataLoader, collate_fn=batch_collate_function)

        if self.train_sampler:
            log.info(self.train_sampler)

        self._train_loader = dataloader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle and not self.train_sampler,
            num_workers=num_workers,
            sampler=self.train_sampler,
        )

        self._test_loader = dataloader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=self.test_sampler,
        )

        if self.val_dataset:
            self._val_loader = dataloader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                sampler=self.val_sampler,
            )

        if precompute_multi_scale:
            self.set_strategies(model)

    @property
    def has_val_loader(self):
        try:
            _ = getattr(self, "_val_loader")
            return True
        except:
            False

    def val_dataloader(self):
        return self._val_loader

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
        return self.train_dataset.num_classes

    @property
    def weight_classes(self):
        return getattr(self.train_dataset, "weight_classes", None)

    @property
    def feature_dimension(self):
        return self.train_dataset.num_features

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_batches(self):
        return {
            "train": len(self._train_loader),
            "test": len(self._test_loader),
            "val": len(self._val_loader) if self.has_val_loader else 0,
        }

    def _set_multiscale_transform(self, transform):
        for _, attr in self.__dict__.items():
            if isinstance(attr, torch.utils.data.DataLoader):
                current_transform = getattr(attr.dataset, "transform", None)
                if current_transform is None:
                    setattr(attr.dataset, "transform", transform)
                else:
                    if isinstance(current_transform, Compose):  # The transform contains several transformations
                        current_transform.transforms += [transform]
                    else:
                        setattr(
                            attr.dataset, "transform", Compose([current_transform, transform]),
                        )

    def set_strategies(self, model):
        strategies = model.get_spatial_ops()

        transform = MultiScaleTransform(strategies)
        self._set_multiscale_transform(transform)

    @staticmethod
    @abstractmethod
    def get_tracker(model, dataset, wandb_log: bool, tensorboard_log: bool):
        pass

    def __repr__(self):
        message = "Dataset: %s \n" % self.__class__.__name__
        for attr in self.__dict__:
            if "transform" in attr:
                message += "{}{} {}= {}\n".format(COLORS.IPurple, attr, COLORS.END_NO_TOKEN, getattr(self, attr))
        for attr in self.__dict__:
            if attr.endswith("_dataset"):
                dataset = getattr(self, attr)
                if dataset:
                    size = len(dataset)
                else:
                    size = 0
                message += "Size of {}{} {}= {}\n".format(COLORS.IPurple, attr, COLORS.END_NO_TOKEN, size)
        for key, attr in self.__dict__.items():
            if key.endswith("_sampler") and attr:
                message += "{}{} {}= {}\n".format(COLORS.IPurple, key, COLORS.END_NO_TOKEN, attr)
        message += "{}Batch size ={} {}".format(COLORS.IPurple, COLORS.END_NO_TOKEN, self.batch_size)
        return message

