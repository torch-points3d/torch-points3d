import torch
from torch_geometric.data import Data, Batch

from torch_points3d.datasets.batch import SimpleBatch
from torch_points3d.core.data_transform import MultiScaleTransform
from torch_points3d.datasets.multiscale_data import MultiScaleBatch
from torch_points3d.datasets.registration.pair import Pair, PairBatch, PairMultiScaleBatch, DensePairBatch

class MockDatasetConfig(object):
    def __init__(self):
        pass

    def keys(self):
        return []

    def get(self, dataset_name, default):
        return None


class MockDataset(torch.utils.data.Dataset):
    def __init__(self, feature_size=0, transform=None, num_points=100):
        self.feature_dimension = feature_size
        self.num_classes = 10
        self.num_points = num_points
        self.batch_size = 2
        self.weight_classes = None
        if feature_size > 0:
            self._feature = torch.tensor([range(feature_size) for i in range(self.num_points)], dtype=torch.float,)
        else:
            self._feature = None
        self._y = torch.tensor([0 for i in range(self.num_points)], dtype=torch.long)
        self._category = torch.ones((self.num_points,), dtype=torch.long)
        self._ms_transform = None
        self._transform = transform

    def __len__(self):
        return self.num_points

    @property
    def datalist(self):
        torch.manual_seed(0)
        datalist = [
            Data(pos=torch.randn((self.num_points, 3)), x=self._feature, y=self._y, category=self._category)
            for i in range(self.batch_size)
        ]
        if self._transform:
            datalist = [self._transform(d.clone()) for d in datalist]
        if self._ms_transform:
            datalist = [self._ms_transform(d.clone()) for d in datalist]
        return datalist

    def __getitem__(self, index):
        return SimpleBatch.from_data_list(self.datalist)

    @property
    def class_to_segments(self):
        return {"class1": [0, 1, 2, 3, 4, 5], "class2": [6, 7, 8, 9]}

    def set_strategies(self, model):
        strategies = model.get_spatial_ops()
        transform = MultiScaleTransform(strategies)
        self._ms_transform = transform


class MockDatasetGeometric(MockDataset):
    def __getitem__(self, index):
        if self._ms_transform:
            return MultiScaleBatch.from_data_list(self.datalist)
        else:
            return Batch.from_data_list(self.datalist)


class PairMockDataset(MockDataset):

    def __init__(self, feature_size=0, transform=None, num_points=100, is_pair_ind=True):
        super(PairMockDataset, self).__init__(feature_size, transform, num_points)
        if(is_pair_ind):
            self._pair_ind = torch.tensor([[0, 1], [1, 0]])
        else:
            self._pair_ind = None

    @property
    def datalist(self):
        torch.manual_seed(0)
        datalist_source = [
            Data(pos=torch.randn((self.num_points, 3)), x=self._feature,
                 pair_ind=self._pair_ind, size_pair_ind=torch.tensor([len(self._pair_ind)]))
            for i in range(self.batch_size)
        ]
        datalist_target = [
            Data(pos=torch.randn((self.num_points, 3)), x=self._feature,
                 pair_ind=self._pair_ind, size_pair_ind=torch.tensor([len(self._pair_ind)]))
            for i in range(self.batch_size)
        ]
        if self._transform:
            datalist_source = [self._transform(d.clone()) for d in datalist_source]
            datalist_target = [self._transform(d.clone()) for d in datalist_target]
        if self._ms_transform:
            datalist_source = [self._ms_transform(d.clone()) for d in datalist_source]
            datalist_target = [self._ms_transform(d.clone()) for d in datalist_target]
        datalist = [Pair.make_pair(datalist_source[i], datalist_target[i]) for i in range(self.batch_size)]
        return datalist

    def __getitem__(self, index):
        return DensePairBatch.from_data_list(self.datalist)


class PairMockDatasetGeometric(PairMockDataset):

    def __getitem__(self, index):

        if self._ms_transform:
            return PairMultiScaleBatch.from_data_list(self.datalist)
        else:
            return PairBatch.from_data_list(self.datalist)
