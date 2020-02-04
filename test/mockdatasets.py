import torch
from torch_geometric.data import Data, Batch

from src.datasets.batch import SimpleBatch
from src.core.data_transform import MultiScaleTransform
from src.datasets.multiscale_data import MultiScaleBatch


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
        self._y = torch.tensor([0 for i in range(self._pos.shape[0])], dtype=torch.long)
        self._batch = torch.tensor(
            [0 for i in range(self._pos.shape[0] // 2)]
            + [1 for i in range(self._pos.shape[0] // 2, self._pos.shape[0])]
        )
        self._category = torch.ones((nb_points,), dtype=torch.long)
        self._transform = None

    @property
    def datalist(self):
        datalist = [Data(pos=self._pos, x=self._feature, y=self._y, category=self._category) for i in range(2)]
        if self._transform:
            datalist = [self._transform(d.clone()) for d in datalist]
        return datalist

    def __getitem__(self, index):
        return SimpleBatch.from_data_list(self.datalist)

    @property
    def class_to_segments(self):
        return {"class1": [0, 1, 2, 3, 4, 5], "class2": [6, 7, 8, 9]}

    def set_strategies(self, model):
        strategies = model.get_sampling_and_search_strategies()
        transform = MultiScaleTransform(strategies)
        self._transform = transform


class MockDatasetGeometric(MockDataset):
    def __getitem__(self, index):
        if self._transform:
            return MultiScaleBatch.from_data_list(self.datalist)
        else:
            return Batch.from_data_list(self.datalist)
