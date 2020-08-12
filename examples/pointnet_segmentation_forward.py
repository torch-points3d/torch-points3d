from omegaconf import OmegaConf
import torch

from utils import load_local_torchpoints3d
load_local_torchpoints3d()

from torch_points3d.models.segmentation.pointnet import PointNet
from torch_geometric.data import Data, Batch
from torch_points3d.datasets.batch import SimpleBatch

##################### PARTIAL_DENSE FORMAT #####################

num_points = 500
num_classes = 10
input_nc = 3

pos = torch.randn((num_points, 3))
x = torch.randn((num_points, input_nc))

data = Data(pos=pos, x=x)
data = Batch.from_data_list([data, data])

print(data)
#Batch(batch=[1000], pos=[1000, 3], x=[1000, 3])

pointnet = PointNet(OmegaConf.create({'conv_type': 'PARTIAL_DENSE'}))

pointnet.set_input(data, "cpu")
data_out = pointnet.forward()
print(data_out.shape)
# torch.Size([1000, 4])

##################### DENSE FORMAT #####################

num_points = 500
num_classes = 10
input_nc = 3

pos = torch.randn((num_points, 3))
x = torch.randn((num_points, input_nc))

data = Data(pos=pos, x=x)
data = SimpleBatch.from_data_list([data, data])

print(data)
#SimpleBatch(pos=[2, 500, 3], x=[2, 500, 3])

pointnet = PointNet(OmegaConf.create({'conv_type': 'DENSE'}))

pointnet.set_input(data, "cpu")
data_out = pointnet.forward()
print(data_out.shape)
#torch.Size([2, 500, 4])