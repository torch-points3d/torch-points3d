import os
import sys
import logging
import importlib

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../..")
sys.path.insert(0, ROOT)

log = logging.getLogger(__name__)

# Import torchsparse for documentation and linting purposes
try:
    from .torchsparse import *  # type: ignore
except:
    try:
        from .minkowski import *  # type: ignore
    except:
        pass


__all__ = ["cat", "Conv3d", "Conv3dTranspose", "ReLU", "SparseTensor", "BatchNorm"]
for val in __all__:
    exec(val + "=None")

def backend_valid(_backend):
    return _backend in {"torchsparse", "minkowski"}

sp3d_backend = None

def get_backend():
    return sp3d_backend

def set_backend(_backend):
    """ Use this method to switch sparse backend dynamically. When importing this module with a wildcard such as
    from torch_points3d.modules.SparseConv3d.nn import *
    make sure that you import it again after calling this method.


    Parameters
    ----------
    backend : str
        "torchsparse" or "minkowski"
    """
    assert backend_valid(_backend)
    try:
        modules = importlib.import_module("." + _backend, __name__)  # noqa: F841
        global sp3d_backend
        sp3d_backend = _backend
    except:
        log.exception("Could not import %s backend for sparse convolutions" % _backend)
    for val in __all__:
        exec("globals()['%s'] = modules.%s" % (val, val))
