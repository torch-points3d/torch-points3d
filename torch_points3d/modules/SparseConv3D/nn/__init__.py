import os

if "SPARSE_BACKEND" in os.environ:
    _backend = os.environ["SPARSE_BACKEND"]
    assert _backend in {"torchsparse", "minkowski"}
    _BACKEND = _backend
else:
    _BACKEND = "torchsparse"

if _BACKEND == "torchsparse":
    from .torchsparse import *
elif _BACKEND == "minkowski":
    from .minkowski import *
else:
    raise ValueError("Backend %s is not supported" % _BACKEND)


__all__ = ["cat", "Conv3d", "Conv3dTranspose", "ReLU", "SparseTensor", "BatchNorm"]
