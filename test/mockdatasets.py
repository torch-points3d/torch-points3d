import torch
from torch_geometric.data import Data, Batch

from datasets.batch import SimpleBatch


class MockDataset(torch.utils.data.Dataset):
    def __init__(self, feature_size=0):
        self.feature_dimension = feature_size
        self.num_classes = 10
        self.weight_classes = None
        nb_points = 100
        self._pos = torch.randn((nb_points, 3))
        if feature_size > 0:
            self._feature = torch.tensor([range(feature_size) for i in range(self._pos.shape[0])], dtype=torch.float,)
        else:
            self._feature = None
        self._y = torch.tensor([range(10) for i in range(self._pos.shape[0])], dtype=torch.float)
        self._batch = torch.tensor([0 for i in range(self._pos.shape[0])])

    def __getitem__(self, index):
        datalist = [Data(pos=self._pos, x=self._feature, y=self._y) for i in range(2)]
        return SimpleBatch.from_data_list(datalist)

    @property
    def class_to_segments(self):
        return {"class1": [0, 1, 2, 3, 4, 5], "class2": [6, 7, 8, 9]}


class MockDatasetGeometric(MockDataset):
    def __getitem__(self, index):
        return Batch(pos=self._pos, x=self._feature, y=self._y, batch=self._batch)
