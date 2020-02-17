import importlib
import torch
import copy
from omegaconf import OmegaConf
from src.datasets.base_dataset import BaseDataset
from src.models.base_model import BaseModel


def contains_key(opt, key):
    try:
        _ = opt[key]
        return True
    except:
        return False


def instantiate_dataset(dataset_class, task, dataset_config, cfg_training):
    """Import the module "data/[module].py".
    In the file, the class called {class_name}() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
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

    dataset = dataset_cls(dataset_config, cfg_training)
    return dataset


def instantiate_model(model_class, task, option, dataset: BaseDataset) -> BaseModel:
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

    option_container = OmegaConf.to_container(option)
    model = model_cls(option, "dummy", dataset, modellib)

    model_state = {"model_class": model_class, "option": option_container, "model_module": model_module, "task": task}
    model.model_state = model_state
    return model
