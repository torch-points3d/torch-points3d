import torch
from torch_geometric.data import Data, Batch

from src.datasets.batch import SimpleBatch
from src.core.data_transform import MultiScaleTransform
from src.datasets.multiscale_data import MultiScaleBatch


class MockDataset(torch.utils.data.Dataset):
    def __init__(self, feature_size=0):
        self.feature_dimension = feature_size
        self.num_classes = 10
        self.num_points = 100
        self.batch_size = 2
        self.weight_classes = None
        if feature_size > 0:
            self._feature = torch.tensor([range(feature_size) for i in range(self.num_points)], dtype=torch.float,)
        else:
            self._feature = None
        self._y = torch.tensor([0 for i in range(self.num_points)], dtype=torch.long)
        self._category = torch.ones((self.num_points,), dtype=torch.long)
        self._transform = None

    @property
    def datalist(self):
        torch.manual_seed(0)
        pos = torch.randn((self.num_points, 3))
        datalist = [
            Data(pos=torch.randn((self.num_points, 3)), x=self._feature, y=self._y, category=self._category)
            for i in range(self.batch_size)
        ]
        if self._transform:
            datalist = [self._transform(d.clone()) for d in datalist]
        return datalist

    def __getitem__(self, index):
        return SimpleBatch.from_data_list(self.datalist)

    @property
    def class_to_segments(self):
        return {"class1": [0, 1, 2, 3, 4, 5], "class2": [6, 7, 8, 9]}

    def set_strategies(self, model):
        strategies = model.get_spatial_ops()
        transform = MultiScaleTransform(strategies)
        self._transform = transform

class MockDatasetGeometric(MockDataset):
    def __getitem__(self, index):
        if self._transform:
            return MultiScaleBatch.from_data_list(self.datalist)
        else:
            return Batch.from_data_list(self.datalist)

