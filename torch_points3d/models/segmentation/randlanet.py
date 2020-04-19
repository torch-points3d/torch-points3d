from .base import Segmentation_MP
from torch_points3d.modules.RandLANet import *


class RandLANetSeg(Segmentation_MP):
    """ Unet base implementation of RandLANet
    """
