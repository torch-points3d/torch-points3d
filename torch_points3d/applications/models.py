from .kpconv import KPConv
from .pointnet2 import PointNet2
from .rsconv import RSConv

try:
    from .sparseconv3d import SparseConv3d
except:
    import logging

    log = logging.getLogger(__name__)
    log.warning(
        "Sparse convolutions are not supported, please install one of the available backends, MinkowskiEngine or MIT SparseConv"
    )
