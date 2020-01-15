import os
import numpy as np
import torch
from torch_geometric.datasets import S3DIS
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

from .base_dataset import BaseDataset


class S3DIS_With_Weights(S3DIS):
    def __init__(
        self,
        root,
        test_area=6,
        train=True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        class_weight_method=None,
    ):
        super(S3DIS_With_Weights, self).__init__(
            root,
            test_area=test_area,
            train=train,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        inv_class_map = {
            0: "ceiling",
            1: "floor",
            2: "wall",
            3: "column",
            4: "beam",
            5: "window",
            6: "door",
            7: "table",
            8: "chair",
            9: "bookcase",
            10: "sofa",
            11: "board",
            12: "clutter",
        }
        if train:
            if class_weight_method is None:
                weights = torch.ones((len(inv_class_map.keys())))
            else:
                self.idx_classes, weights = torch.unique(
                    self.data.y, return_counts=True
                )
                weights = weights.float()
                weights = weights.mean() / weights
                if class_weight_method == "sqrt":
                    weights = torch.sqrt(weights)
                elif str(class_weight_method).startswith("log"):
                    w = float(class_weight_method.replace("log", ""))
                    weights = 1 / torch.log(1.1 + weights / weights.sum())

                weights /= torch.sum(weights)
            print(
                "CLASS WEIGHT : {}".format(
                    {
                        name: np.round(weights[index].item(), 4)
                        for index, name in inv_class_map.items()
                    }
                )
            )
            self.weight_classes = weights
        else:
            self.weight_classes = torch.ones((len(inv_class_map.keys())))


class S3DIS1x1Dataset(BaseDataset):
    def __init__(self, dataset_opt, training_opt):
        super().__init__(dataset_opt, training_opt)
        self._data_path = os.path.join(dataset_opt.dataroot, "S3DIS1x1")

        transform = T.Compose(
            [
                T.FixedPoints(dataset_opt.num_points),
                T.RandomTranslate(0.01),
                T.RandomRotate(180, axis=2),
            ]
        )
        train_dataset = S3DIS_With_Weights(
            self._data_path,
            test_area=self.dataset_opt.fold,
            train=True,
            transform=transform,
            class_weight_method=dataset_opt.class_weight_method,
        )
        test_dataset = S3DIS_With_Weights(
            self._data_path,
            test_area=self.dataset_opt.fold,
            train=False,
            transform=T.FixedPoints(dataset_opt.num_points),
        )

        self._create_dataloaders(train_dataset, test_dataset, validation=None)
