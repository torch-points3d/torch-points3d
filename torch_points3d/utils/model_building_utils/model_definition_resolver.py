from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig


def resolve_model(model_config, dataset, tested_task):
    """ Parses the model config and evaluates any expression that may contain constants
    """
    # placeholders to subsitute
    constants = {
        "FEAT": max(dataset.feature_dimension, 0),
        "TASK": tested_task,
        "N_CLS": dataset.num_classes if hasattr(dataset, "num_classes") else None,
    }

    # user defined contants to subsitute
    if "define_constants" in model_config.keys():
        constants.update(dict(model_config.define_constants))

    resolve(model_config, constants)


def resolve(obj, constants):
    """ Resolves expressions and constants in obj.
    returns False if obj is a ListConfig or DictConfig, True is obj is a primative type.
    """
    if type(obj) == DictConfig:
        it = (k for k in obj)
    elif type(obj) == ListConfig:
        it = range(len(obj))
    else:
        # obj is a single element
        return True

    # recursively resolve all children of obj
    for k in it:

        # if obj[k] is a primative type, evalulate it
        if resolve(obj[k], constants):
            if type(obj[k]) is str:
                try:
                    obj[k] = eval(obj[k], constants)
                except NameError:
                    # we tried to resolve a string which isn't an expression
                    pass
                except ValueError:
                    # we tried to resolve a string which is also a builtin (e.g. max)
                    pass
                except Exception as e:
                    print(e)

    return False
