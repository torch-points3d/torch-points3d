import importlib
from src.datasets.base_dataset import BaseDataset
from src.architectures.base_model import BaseModel


def contains_key(opt, key):
    try:
        _ = opt[key]
        return True
    except:
        return False


def find_dataset_using_name(dataset_name, tested_task):
    """Import the module "data/[dataset_name]_dataset.py".
    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """

    dataset_filename = "src.datasets.{}.{}_dataset".format(tested_task, dataset_name)
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


def find_model_using_name(model_logic, model_type, task, option, dataset: BaseDataset) -> BaseModel:

    assert getattr(model_logic, "file_name") != None, "The model need to contain a logic_model with a file_name"
    assert getattr(model_logic, "class_name") != None, "The model need to contain a logic_model with a class_name"

    model_filename = ".".join(["src.models", task, model_logic.file_name])
    modellib = importlib.import_module(model_filename)

    for name, cls in modellib.__dict__.items():
        if name.lower() == model_logic.class_name.lower():
            model = cls

    if model is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase."
            % (model_type, task)
        )

    # module_filename = ".".join(["src.modules", model_type, "modules"])
    # modules_lib = importlib.import_module(module_filename)
    return model(option, model_type, dataset, modellib)
