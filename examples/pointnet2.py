import torch
from torch_points3d.applications.pointnet2 import PointNet2
from torch_geometric.data import Batch, Data

num_points = 1024
num_classes = 10
input_nc = 5

pos = torch.randn((num_points, 3)).unsqueeze(0)
T = torch.randn((num_points, input_nc)).unsqueeze(0)


data = Data(pos=pos, x=T)

# Batch(batch=[2], pos=[2, 1024, 3], x=[2, 1024, 5])
data = Batch.from_data_list([data, data])

model = PointNet2(architecture="encoder", input_nc=input_nc, num_layers=3, output_nc=num_classes,)

# Data(x=[2, 10, 1])
res = model(data)
