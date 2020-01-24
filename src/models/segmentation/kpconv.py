from .base import Segmentation_MP
from src.modules.KPConv import *
from src.models.base_model import BaseModel


class KPConvSeg(Segmentation_MP):
    """ Basic implementation of KPConv"""

    def set_input(self, data):
        self.input = data
        self.batch_idx = data.batch
        self.labels = data.y
