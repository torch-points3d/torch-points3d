import importlib
from datasets.base_dataset import BaseDataset
from architectures.base_model import BaseModel

def contains_key(opt, key):
    try:
        _ = opt[key]
        return True
    except:
        return False


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".
    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "src.datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace("_", "") + "dataset"
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase."
            % (dataset_filename, target_dataset_name)
        )

    return dataset


def find_model_using_name(model_type, task, option, dataset: BaseDataset) -> BaseModel:

    model_filename = ".".join(["models", model_type, "nn"])
    modellib = importlib.import_module(model_filename)

    for name, cls in modellib.__dict__.items():
        if name.lower() == cls_name.lower():
            model = cls

    if model is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase."
            % (model_type, task)
        )

    module_filename = ".".join(["models", model_type, "modules"])
    modules_lib = importlib.import_module(module_filename)
    return model(option, model_type, dataset, modules_lib)