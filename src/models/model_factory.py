import importlib
import hydra

from src.datasets.base_dataset import BaseDataset
from .base_model import BaseModel
from src.utils.model_building_utils.model_definition_resolver import resolve_model


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
