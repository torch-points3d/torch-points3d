from .kpconv import KPConv
from .pointnet2 import PointNet2
from .rsconv import RSConv
import logging

log = logging.getLogger(__name__)

try:
    from .sparseconv3d import SparseConv3d
except:
    log.warning(
        "Sparse convolutions are not supported, please install one of the available backends, MinkowskiEngine or MIT SparseConv"
    )

try:
    from .minkowski import Minkowski
except:
    log.warning("MinkowskiEngine is not installed.")
