from .base import Segmentation_MP
from src.modules.PointCNN import *


class PointCNNSeg(Segmentation_MP):
    """ Unet base implementation of PointCNN
    https://arxiv.org/abs/1801.07791
    """
