import torch.nn as nn
import MinkowskiEngine as ME

from .modules import SparseResBlock
from src.utils.config import is_list


class Sequential(nn.Sequential):
    def __add__(self, x):
        r = Sequential()
        for m in self:
            r.append(m)
        for m in x:
            r.append(m)
        return r

    def add(self, module):
        self._modules[str(len(self._modules))] = module
        return self


class SkipModule(nn.Module):

    CONNECTIONS = ["none", "residual", "concat"]

    def __init__(self, submodule, connection="concat"):
        super(SkipModule, self).__init__()

        if connection.lower() not in self.CONNECTIONS:
            raise Exception("connection {} not in {}".format(connection, self.CONNECTIONS))
        self._submodule = submodule
        self.connection = connection.lower()

    def forward(self, skip_x):
        if self.connection == "concat":
            return ME.cat(skip_x, self._submodule(skip_x))
        elif self.connection == "residual":
            return skip_x + self._submodule(skip_x)
        else:
            return self._submodule(skip_x)

    def add(self, module):
        self._modules[str(len(self._modules))] = module
        return self


class UnwrappedSparseUnet(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        input_planes=[8, 16, 32, 64, 128],
        norm_layer="MinkowskiBatchNorm",
        use_dropout=False,
        n_reps=1,
        dim=3,
        dropout_rate=0.1,
        mix_conv=False,
    ):
        super(UnwrappedSparseUnet, self).__init__()
        self.dimension = dim
        self.reps = n_reps
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.downsample = [2, 2]
        self.leakiness = False
        self.norm_layer = getattr(ME, norm_layer, None)
        self.mix_conv = mix_conv

        modules = []

        modules.append(ME.MinkowskiConvolution(input_nc, 8, kernel_size=3, dimension=self.dimension))
        modules.append(self.U(input_planes))
        modules.append(self.norm_layer(8))
        modules.append(ME.MinkowskiConvolution(8, output_nc, kernel_size=3, dimension=self.dimension))
        modules.append(ME.MinkowskiTanh())

        self.model = nn.Sequential(*modules)

    def U(self, nPlanes, n_input_planes=-1):  # Recursive function
        m = Sequential()
        for i in range(self.reps):
            m.add(
                SparseResBlock(
                    n_input_planes if n_input_planes != -1 else nPlanes[0],
                    nPlanes[0],
                    self.dimension,
                    use_dropout=self.use_dropout,
                    dropout_rate=self.dropout_rate,
                    norm_layer=self.norm_layer,
                    mix_conv=self.mix_conv,
                )
            )
            n_input_planes = -1

        if len(nPlanes) > 1:

            residual = (
                Sequential()
                .add(self.norm_layer(nPlanes[0]))
                .add(ME.MinkowskiConvolution(nPlanes[0], nPlanes[1], kernel_size=3, stride=2, dimension=self.dimension))
                .add(self.U(nPlanes[1:]))
                .add(self.norm_layer(nPlanes[1]))
                .add(
                    ME.MinkowskiConvolutionTranspose(
                        nPlanes[1], nPlanes[0], kernel_size=3, stride=2, dimension=self.dimension
                    )
                )
            )

            m.add(SkipModule(residual))

            for i in range(self.reps):
                m.add(
                    SparseResBlock(
                        nPlanes[0] * (2 if i == 0 else 1),
                        nPlanes[0],
                        self.dimension,
                        use_dropout=self.use_dropout,
                        dropout_rate=self.dropout_rate,
                        norm_layer=self.norm_layer,
                        mix_conv=self.mix_conv,
                    )
                )
        return m

    def forward(self, x):
        x = self.model(x)
        return x
