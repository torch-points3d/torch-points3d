from models.base_model import BaseConvolution, MLP
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils import remove_self_loops, add_self_loops
import torch
from torch_geometric.nn import PointConv

class SAModule(BaseConvolution):
    def __init__(self, ratio=None, radius=None, down_conv_nn=None, *args, **kwargs):
        super(SAModule, self).__init__(ratio, radius)      

        local_nn = MLP(down_conv_nn) if down_conv_nn is not None else None
        self._conv = PointConv(local_nn=local_nn, global_nn=None)

    @property
    def conv(self):
        return self._conv