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
from src.utils.colors import COLORS, colored_print
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

        if isinstance(self.test_dataset, list):
            self._test_loaders = [dataloader(dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, sampler=self.test_sampler,
                                ) for dataset in self.test_dataset]
        else:
            self._test_loaders = [dataloader(
                self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=self.test_sampler,
            )]
        self._test_dataset_names = [self.get_test_dataset_name(idx) for idx in range(self.num_test_datasets)]

        if self.num_test_datasets > 1:
            log.info(" ")
            colored_print(COLORS.Yellow, 'You are using multiple test sets, choose amongst {} by setting selection_stage to any of them. If you want to have better trackable names, add a "dataset_name" attributes to the dataset. Could be done on either one, couple or any'.format(self._test_dataset_names))

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

    def test_dataloaders(self):
        return self._test_loaders

    @property
    def num_test_datasets(self):
        return len(self._test_loaders)

    @property
    def test_datatset_names(self):
        return self._test_dataset_names

    @property
    def has_fixed_points_transform(self):
        """
        This property checks if the dataset contains T.FixedPoints transform, meaning the number of points is fixed
        """
        transform_train = self._train_loader.dataset.transform
        transform_test = self._test_loaders.dataset.transform

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

    def get_test_dataset_name(self, index=None):
        loader = self._test_loaders[index]
        if hasattr(loader.dataset, "dataset_name"):
            return loader.dataset.dataset_name
        else:
            if self.num_test_datasets > 1:
                return "test_{}".format(index)
            else:
                return"test"

    @property
    def determine_stage(self):
        if self.has_val_loader:
            return "val"
        else:
            return self.get_test_dataset_name(0)

    @property
    def num_batches(self):
        out = {
            "train": len(self._train_loader),
            "val": len(self._val_loader) if self.has_val_loader else 0,
        }
        stage_name = "test"
        if len(self._test_loaders) > 1:
            for loader_idx, loader in enumerate(self._test_loaders):
                stage_name = getattr(loader, "dataset_name", None)
                if stage_name is None:
                    stage_name = "test:{}".format(loader_idx)
                out[stage_name] = len(loader)
        else:
            out[stage_name] = len(self._test_loaders[0])
        return out

    def _set_composed_multiscale_transform(self, attr, transform):
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

    def _set_multiscale_transform(self, transform):
        for attr_name, attr in self.__dict__.items():
            if isinstance(attr, torch.utils.data.DataLoader):
                self._set_composed_multiscale_transform(attr, transform)

        for loader in self._test_loaders:
            self._set_composed_multiscale_transform(loader, transform)

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
                if isinstance(dataset, list):
                    if len(dataset) > 1:
                        size = ", ".join([str(len(d)) for d in dataset])
                    else:
                        size = len(dataset[0])
                elif dataset:
                    size = len(dataset)
                else:
                    size = 0
                message += "Size of {}{} {}= {}\n".format(COLORS.IPurple, attr, COLORS.END_NO_TOKEN, size)
        for key, attr in self.__dict__.items():
            if key.endswith("_sampler") and attr:
                message += "{}{} {}= {}\n".format(COLORS.IPurple, key, COLORS.END_NO_TOKEN, attr)
        message += "{}Batch size ={} {}".format(COLORS.IPurple, COLORS.END_NO_TOKEN, self.batch_size)
        return message

