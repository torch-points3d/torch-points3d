import os
import importlib

def find_model_using_name(model_name, task, option, num_classes):

    if task == "SEGMENTATION":
        cls_name = "SegmentationModel"
    else:
        cls_name = "ClassificationModel"

    dataset_filename = '.'.join(["models", model_name, "nn"])
    datasetlib = importlib.import_module(dataset_filename)

    for name, cls in datasetlib.__dict__.items():
        if name.lower() == cls_name.lower():
            model = cls

    if model is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (
            model_name, task))
    
    module_filename = '.'.join(["models", model_name, "modules"])
    modules = importlib.import_module(module_filename)
    return model(option, num_classes, modules)