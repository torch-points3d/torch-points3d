import importlib
import copy
import hydra
import logging
from contextlib import contextmanager
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_points3d.datasets.base_dataset import BaseDataset

log = logging.getLogger(__name__)


def get_dataset_class(dataset_config):
    task = dataset_config.task
    # Find and create associated dataset
    try:
        dataset_config.dataroot = hydra.utils.to_absolute_path(
            dataset_config.dataroot)
    except Exception:
        log.error("This should happen only during testing")
    dataset_class = getattr(dataset_config, "class")
    dataset_paths = dataset_class.split(".")
    module = ".".join(dataset_paths[:-1])
    class_name = dataset_paths[-1]
    dataset_module = ".".join(["torch_points3d.datasets", task, module])
    datasetlib = importlib.import_module(dataset_module)

    target_dataset_name = class_name
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset_cls = cls

    if dataset_cls is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase."
            % (module, class_name)
        )
    return dataset_cls


def instantiate_dataset(dataset_config) -> BaseDataset:
    """Import the module "data/[module].py".
    In the file, the class called {class_name}() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_cls = get_dataset_class(dataset_config)
    dataset = dataset_cls(dataset_config)
    return dataset

class LiftDataModule(LightningDataModule):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.trackers = {}

        for key, value in vars(dataset).items():
            setattr(self, key, value)

        for stage in ["train", "val", "test"]:
            self.add_loader(stage)

    def has_loader(self, stage):
        loader = "loader"
        if stage == "test": loader = "loaders"
        return getattr(self, "_{stage}_{loader}", None) is None

    def add_loader(self, stage):
        if not self.has_loader(stage):
            setattr(self, f"{stage}_dataloader", None)
        else:
            self.trackers[stage] = self.dataset.get_tracker(False, False)
            self.trackers[stage].reset(stage)

    def  train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._train_loader

    def  val_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._val_loader

    def  test_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._test_loaders

    def __repr__(self):
        return "LightningDataModule \n" + self.dataset.__repr__()


def convert_to_lightning_data_module(dataset):
    return LiftDataModule(dataset)
