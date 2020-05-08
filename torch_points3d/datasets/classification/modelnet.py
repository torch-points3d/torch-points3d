import os.path as osp

from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

from torch_points3d.datasets.base_dataset import BaseDataset

class ModelNetDataset(BaseDataset):

    AVAILABLE_NUMBERS = ["10", "40"]

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        number = dataset_opt.number
        if str(number) not in self.AVAILABLE_NUMBERS:
            raise Exception("Only ModelNet10 and ModelNet40 are available")
        name = "ModelNet{}".format(number)
        self._data_path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", name)
        pre_transform = T.NormalizeScale()
        transform = T.SamplePoints(dataset_opt.num_points) if dataset_opt.num_points else None

        self.train_dataset = ModelNet(
            self._data_path, name=str(number), train=True, transform=transform, pre_transform=pre_transform,
        )

        self.test_dataset = ModelNet(
            self._data_path, name=str(number), train=False, transform=transform, pre_transform=pre_transform,
        )