import importlib
import torch
import copy
from omegaconf import OmegaConf
import hydra
import logging

from src.datasets.base_dataset import BaseDataset
from src.models.base_model import BaseModel
from src.utils.model_building_utils.model_definition_resolver import resolve_model

log = logging.getLogger(__name__)


def contains_key(opt, key):
    try:
        _ = opt[key]
        return True
    except:
        return False


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


def instantiate_model(config, dataset: BaseDataset) -> BaseModel:
    """ Creates a model given a datset and a training config. The config should contain the following:
    - config.data.task: task that will be evaluated
    - config.model_name: model to instantiate
    - config.models: All models available
    """

    # Get task and model_name
    task = config.data.task
    tested_model_name = config.model_name

    # Find configs
    model_config = getattr(config.models, tested_model_name, None)
    resolve_model(model_config, dataset, task)

    model_class = getattr(model_config, "class")
    model_paths = model_class.split(".")
    module = ".".join(model_paths[:-1])
    class_name = model_paths[-1]
    model_module = ".".join(["src.models", task, module])
    modellib = importlib.import_module(model_module)

    for name, cls in modellib.__dict__.items():
        if name.lower() == class_name.lower():
            model_cls = cls

    if model_cls is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase."
            % (model_module, class_name)
        )
    model = model_cls(model_config, "dummy", dataset, modellib)
    return model
