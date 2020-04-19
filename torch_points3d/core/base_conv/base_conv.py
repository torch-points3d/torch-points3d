from abc import ABC

from torch_points3d.core.common_modules.base_modules import BaseModule


class BaseConvolution(ABC, BaseModule):
    def __init__(self, sampler, neighbour_finder, *args, **kwargs):
        BaseModule.__init__(self)
        self.sampler = sampler
        self.neighbour_finder = neighbour_finder
