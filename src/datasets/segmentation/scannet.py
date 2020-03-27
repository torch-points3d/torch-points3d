class ScannetDataset:import os
import os.path as osp
import shutil
import json
from tqdm import tqdm as tq
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_txt_array
import torch_geometric.transforms as T

from src.metrics.shapenet_part_tracker import ShapenetPartTracker

from src.datasets.base_dataset import BaseDataset

class Scannet(InMemoryDataset):

    CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                    'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                    'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
    VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
    SCANNET_COLOR_MAP = {
        0: (0., 0., 0.),
        1: (174., 199., 232.),
        2: (152., 223., 138.),
        3: (31., 119., 180.),
        4: (255., 187., 120.),
        5: (188., 189., 34.),
        6: (140., 86., 75.),
        7: (255., 152., 150.),
        8: (214., 39., 40.),
        9: (197., 176., 213.),
        10: (148., 103., 189.),
        11: (196., 156., 148.),
        12: (23., 190., 207.),
        14: (247., 182., 210.),
        15: (66., 188., 102.),
        16: (219., 219., 141.),
        17: (140., 57., 197.),
        18: (202., 185., 52.),
        19: (51., 176., 203.),
        20: (200., 54., 131.),
        21: (92., 193., 61.),
        22: (78., 71., 183.),
        23: (172., 114., 82.),
        24: (255., 127., 14.),
        25: (91., 163., 138.),
        26: (153., 98., 156.),
        27: (140., 153., 101.),
        28: (158., 218., 229.),
        29: (100., 125., 154.),
        30: (178., 127., 135.),
        32: (146., 111., 194.),
        33: (44., 160., 44.),
        34: (112., 128., 144.),
        35: (96., 207., 209.),
        36: (227., 119., 194.),
        37: (213., 92., 176.),
        38: (94., 106., 211.),
        39: (82., 84., 163.),
        40: (100., 85., 144.),
    }

    def __init__(
        self,
        root,
        split="trainval",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)
        self.categories = categories
        super(ShapeNet, self).__init__(root, transform, pre_transform, pre_filter)

        if split == "train":
            path = self.processed_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
        elif split == "trainval":
            path = self.processed_paths[3]
        else:
            raise ValueError((f"Split {split} found, but expected either " "train, val, trainval or test"))

        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return list(self.category_ids.values()) + ["train_test_split"]

    @property
    def processed_file_names(self):
        cats = "_".join([cat[:3].lower() for cat in self.categories])
        return [os.path.join("{}_{}.pt".format(cats, split)) for split in ["train", "val", "test", "trainval"]]

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        name = self.url.split("/")[-1].split(".")[0]
        os.rename(osp.join(self.root, name), self.raw_dir)

    def process_filenames(self, filenames):
        data_list = []
        categories_ids = [self.category_ids[cat] for cat in self.categories]
        cat_idx = {categories_ids[i]: i for i in range(len(categories_ids))}

        for name in tq(filenames):
            cat = name.split(osp.sep)[0]
            if cat not in categories_ids:
                continue

            data = read_txt_array(osp.join(self.raw_dir, name))
            pos = data[:, :3]
            x = data[:, 3:6]
            y = data[:, -1].type(torch.long)
            category = torch.ones(x.shape[0], dtype=torch.long) * cat_idx[cat]
            data = Data(pos=pos, x=x, y=y, category=category)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list

    def process(self):
        trainval = []
        for i, split in enumerate(["train", "val", "test"]):
            path = osp.join(self.raw_dir, "train_test_split", f"shuffled_{split}_file_list.json")
            with open(path, "r") as f:
                filenames = [
                    osp.sep.join(name.split(osp.sep)[1:]) + ".txt" for name in json.load(f)
                ]  # Removing first directory.
            data_list = self.process_filenames(filenames)
            if split == "train" or split == "val":
                trainval += data_list
            torch.save(self.collate(data_list), self.processed_paths[i])
        torch.save(self.collate(trainval), self.processed_paths[3])

    def __repr__(self):
        return "{}({}, categories={})".format(self.__class__.__name__, len(self), self.categories)


class ShapeNetDataset(BaseDataset):
    FORWARD_CLASS = "forward.shapenet.ForwardShapenetDataset"

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        try:
            self._category = dataset_opt.category
        except KeyError:
            self._category = None
        pre_transform = self.pre_transform
        train_transform = self.train_transform
        self.train_dataset = ShapeNet(
            self._data_path,
            self._category,
            include_normals=dataset_opt.normal,
            split="trainval",
            pre_transform=pre_transform,
            transform=train_transform,
        )

        self.test_dataset = ShapeNet(
            self._data_path,
            self._category,
            include_normals=dataset_opt.normal,
            split="test",
            transform=self.test_transform,
            pre_transform=pre_transform,
        )
        self._categories = self.train_dataset.categories

    @property
    def class_to_segments(self):
        classes_to_segment = {}
        for key in self._categories:
            classes_to_segment[key] = ShapeNet.seg_classes[key]
        return classes_to_segment

    @property
    def is_hierarchical(self):
        return len(self._categories) > 1

    @staticmethod
    def get_tracker(model, dataset, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            dataset {[type]}
            wandb_log - Log using weight and biases
        Returns:
            [BaseTracker] -- tracker
        """
        return ShapenetPartTracker(dataset, wandb_log=wandb_log, use_tensorboard=tensorboard_log)

