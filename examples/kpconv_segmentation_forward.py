import numpy as np
import torch
from torch_points3d.applications.kpconv import KPConv
from torch_geometric.data import Batch, Data

# KPConv is implemented with PARTIAL_DENSE format. Therefore, data need an attribute batch containing the indice for each point

input_nc = 0
num_classes = 10
batch_size = 3
num_points_per_sample = [5, 10, 3]

model = KPConv(
    architecture="unet",  # Could be unet here to perform segmentation
    input_nc=input_nc,  # KPconv is particular. Pos aren't features. It needs a tensor of ones + any features available as rgb or intensity
    output_nc=num_classes,
    num_layers=4,
)

samples = []
for idx_batch in range(batch_size):
    nb_points = num_points_per_sample[idx_batch]
    pos = torch.randn((nb_points, 3))
    y = torch.from_numpy(np.random.choice(range(num_classes), nb_points))
    x = torch.ones((nb_points, 1))
    samples.append(Data(pos=pos, y=y, x=x))
data = Batch.from_data_list(samples)

print(data)
# Batch(batch=[18], pos=[18, 3], x=[18, 1], y=[18])

print(data.batch)
# tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2])

output = model.forward(data)
print(output)
# Batch(batch=[18], block_idx=2, idx_neighboors=[18, 25], pos=[18, 3], x=[18, 10], y=[18])
