from models.base_model import BaseConvolution, PointConv, MLP

class SAModule(BaseConvolution):
    def __init__(self, ratio=None, radius=None, down_conv_nn=None, *args, **kwargs):
        super(SAModule, self).__init__(ratio, radius)      

        local_nn = MLP(down_conv_nn) if down_conv_nn is not None else None
        self.conv = PointConv(local_nn=local_nn, global_nn=None)