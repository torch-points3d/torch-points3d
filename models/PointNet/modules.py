
import torch 
from torch.nn import Linear
from torch_geometric.nn import global_max_pool

from models.core_modules import * 
from models.core_transforms import LinearTransformSTNkD

class MiniPointNet(torch.nn.Module):

    def __init__(self, local_nn, global_nn):
        super(MiniPointNet, self).__init__()

        self.local_nn = MLP(local_nn)
        self.global_nn = MLP(global_nn)

    def forward(self, x, batch):
        
        x = self.local_nn(x)
        x = global_max_pool(x, batch)
        x = self.global_nn(x)

        return x

class PointNetSTN3D(LinearTransformSTNkD):

    def __init__(self, local_nn = [3, 64, 128, 1024], global_nn = [1024, 512, 256], batch_size=1):
        super().__init__(
            MiniPointNet(local_nn, global_nn),
            global_nn[-1],
            3,
            batch_size
        )

    def forward(self, x, batch):
        return super().forward(x, x, batch)




