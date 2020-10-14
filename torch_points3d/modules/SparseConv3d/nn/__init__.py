import os
import logging
import importlib

log = logging.getLogger(__name__)

try:
    import torch_points3d.modules.SparseConv3d.nn.torchsparse as torchsparse_
    from torch_points3d.modules.SparseConv3d.nn.torchsparse import *
except:
    log.error("Could not import torchsparse backend for sparse convolutions")

try:
    import torch_points3d.modules.SparseConv3d.nn.minkowski as minkowski_
    from torch_points3d.modules.SparseConv3d.nn.minkowski import *
except:
    log.error("Could not import Minkowski backend for sparse convolutions")


__all__ = ["cat", "Conv3d", "Conv3dTranspose", "ReLU", "SparseTensor", "BatchNorm"]
for val in __all__:
    exec(val + "=None")


def set_backend(backend):
    """ Use this method to switch sparse backend dynamically. When importing this module with a wildcard such as
    from torch_points3d.modules.SparseConv3d.nn import *
    make sure that you import it again after calling this method.


    Parameters
    ----------
    backend : str
        "torchsparse" or "minkowski"
    """
    assert backend in {"torchsparse", "minkowski"}
    for val in __all__:
        exec("globals()['%s'] = %s_.%s" % (val, backend, val))


if "SPARSE_BACKEND" in os.environ:
    backend = os.environ["SPARSE_BACKEND"]
else:
    backend = "torchsparse"

set_backend(backend)

__all__ = __all__
