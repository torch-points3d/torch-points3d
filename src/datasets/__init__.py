import importlib
import copy
import hydra

from src.datasets.base_dataset import BaseDataset


def instantiate_dataset(dataset_config) -> BaseDataset:
    """Import the module "data/[module].py".
    In the file, the class called {class_name}() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    task = dataset_config.task

    # Find and create associated dataset
    dataset_config.dataroot = hydra.utils.to_absolute_path(dataset_config.dataroot)

    dataset_class = getattr(dataset_config, "class")
    dataset_paths = dataset_class.split(".")
    module = ".".join(dataset_paths[:-1])
    class_name = dataset_paths[-1]
    dataset_module = ".".join(["src.datasets", task, module])
    datasetlib = importlib.import_module(dataset_module)

    dataset = None
    target_dataset_name = class_name
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset_cls = cls

    if dataset_cls is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase."
            % (module, class_name)
        )

    dataset = dataset_cls(dataset_config)
    return dataset
