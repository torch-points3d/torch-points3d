from torch_geometric.nn import fps, radius, knn
import torch

check = torch.load('knn_debug.p')

x = check['x']
y = check['y']
k = check['k']
batch_x = check['batch_x']
batch_y = check['batch_y']

print(x.shape, y.shape, k, batch_x.shape, batch_y.shape)

import pdb
pdb.set_trace()

knn(x, y, k, batch_x, batch_y)
