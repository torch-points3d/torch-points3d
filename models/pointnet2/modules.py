from torch_geometric.nn import PointConv
from models.core_modules import *
from models.core_sampling_and_search import RadiusNeighbourFinder, FPSSampler


class SAModule(BaseConvolutionDown):
    def __init__(self, ratio=None, radius=None, down_conv_nn=None, nb_feature=None, *args, **kwargs):
        super(SAModule, self).__init__(FPSSampler(ratio), RadiusNeighbourFinder(radius), *args, **kwargs)

        local_nn = MLP(down_conv_nn) if down_conv_nn is not None else None
        self._conv = PointConv(local_nn=local_nn, global_nn=None)

    def conv(self, x, pos, edge_index, batch):
        return self._conv(x, pos, edge_index)
