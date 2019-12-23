import os
import importlib
from .base_model import BaseModel
from datasets.base_dataset import BaseDataset

def find_model_using_name(model_type, task, option, dataset: BaseDataset) -> BaseModel:

    if task == "segmentation":
        cls_name = "SegmentationModel"
    else:
        cls_name = "ClassificationModel"

    model_filename = '.'.join(["models", model_type, "nn"])
    modellib = importlib.import_module(model_filename)

    for name, cls in modellib.__dict__.items():
        if name.lower() == cls_name.lower():
            model = cls

    if model is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (
            model_type, task))

    module_filename = '.'.join(["models", model_type, "modules"])
    modules_lib = importlib.import_module(module_filename)
    return model(option, model_type, dataset, modules_lib)

def resolve_mlp_list(mlp, expectedStartDim=None, expectedEndDim=None):

    DYNAMIC_PLACEHOLDER = '_'

    if expectedStartDim is not None:
        if mlp[0] == DYNAMIC_PLACEHOLDER:
            mlp[0] = expectedStartDim
        elif mlp[0] != expectedStartDim:
            raise ValueError("Layer 0 of {} is misshapen".format(mlp))

    if expectedEndDim is not None:
        if mlp[-1] == DYNAMIC_PLACEHOLDER:
            mlp[-1] = expectedEndDim
        elif mlp[-1] != expectedEndDim:
            raise ValueError("Layer -1 of {} is misshapen".format(mlp))

    if any(x == DYNAMIC_PLACEHOLDER for x in mlp):
        raise ValueError("MLP list contains placeholders which cannot be resolved")

    return mlp 

    

    