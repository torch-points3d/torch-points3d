import os
import os.path as osp
import shutil
import json
from tqdm.auto import tqdm as tq
from itertools import repeat, product
import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import read_txt_array
import torch_geometric.transforms as T
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.metrics.shapenet_part_tracker import ShapenetPartTracker
from torch_points3d.datasets.base_dataset import BaseDataset, save_used_properties
from torch_points3d.utils.download import download_url



class ShapeNet(InMemoryDataset):
    r"""The ShapeNet part level segmentation dataset from the `"A Scalable
    Active Framework for Region Annotation in 3D Shape Collections"
    <http://web.stanford.edu/~ericyi/papers/part_annotation_16_small.pdf>`_
    paper, containing about 17,000 3D shape point clouds from 16 shape
    categories.
    Each category is annotated with 2 to 6 parts.

    Args:
        root (string): Root directory where the dataset should be saved.
        categories (string or [string], optional): The category of the CAD
            models (one or a combination of :obj:`"Airplane"`, :obj:`"Bag"`,
            :obj:`"Cap"`, :obj:`"Car"`, :obj:`"Chair"`, :obj:`"Earphone"`,
            :obj:`"Guitar"`, :obj:`"Knife"`, :obj:`"Lamp"`, :obj:`"Laptop"`,
            :obj:`"Motorbike"`, :obj:`"Mug"`, :obj:`"Pistol"`, :obj:`"Rocket"`,
            :obj:`"Skateboard"`, :obj:`"Table"`).
            Can be explicitly set to :obj:`None` to load all categories.
            (default: :obj:`None`)
        include_normals (bool, optional): If set to :obj:`False`, will not
            include normal vectors as input features. (default: :obj:`True`)
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"trainval"`, loads the training and validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"trainval"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = "https://shapenet.cs.stanford.edu/media/" "shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"

    category_ids = {
        "Airplane": "02691156",
        "Bag": "02773838",
        "Cap": "02954340",
        "Car": "02958343",
        "Chair": "03001627",
        "Earphone": "03261776",
        "Guitar": "03467517",
        "Knife": "03624134",
        "Lamp": "03636649",
        "Laptop": "03642806",
        "Motorbike": "03790512",
        "Mug": "03797390",
        "Pistol": "03948459",
        "Rocket": "04099429",
        "Skateboard": "04225987",
        "Table": "04379243",
    }

    seg_classes = {
        "Airplane": [0, 1, 2, 3],
        "Bag": [4, 5],
        "Cap": [6, 7],
        "Car": [8, 9, 10, 11],
        "Chair": [12, 13, 14, 15],
        "Earphone": [16, 17, 18],
        "Guitar": [19, 20, 21],
        "Knife": [22, 23],
        "Lamp": [24, 25, 26, 27],
        "Laptop": [28, 29],
        "Motorbike": [30, 31, 32, 33, 34, 35],
        "Mug": [36, 37],
        "Pistol": [38, 39, 40],
        "Rocket": [41, 42, 43],
        "Skateboard": [44, 45, 46],
        "Table": [47, 48, 49],
    }

    def __init__(
        self,
        root,
        categories=None,
        include_normals=True,
        split="trainval",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        is_test=False,
    ):
        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)
        self.categories = categories
        self.is_test = is_test
        super(ShapeNet, self).__init__(
            root, transform, pre_transform, pre_filter)

        if split == "train":
            path = self.processed_paths[0]
            raw_path = self.processed_raw_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
            raw_path = self.processed_raw_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
            raw_path = self.processed_raw_paths[2]
        elif split == "trainval":
            path = self.processed_paths[3]
            raw_path = self.processed_raw_paths[3]
        else:
            raise ValueError(
                (f"Split {split} found, but expected either " "train, val, trainval or test"))

        self.data, self.slices, self.y_mask = self.load_data(
            path, include_normals)

        # We have perform a slighly optimzation on memory space of no pre-transform was used.
        # c.f self._process_filenames
        if os.path.exists(raw_path):
            self.raw_data, self.raw_slices, _ = self.load_data(
                raw_path, include_normals)
        else:
            self.get_raw_data = self.get

    def load_data(self, path, include_normals):
        '''This function is used twice to load data for both raw and pre_transformed
        '''
        data, slices = torch.load(path)
        data.x = data.x if include_normals else None

        y_mask = torch.zeros(
            (len(self.seg_classes.keys()), 50), dtype=torch.bool)
        for i, labels in enumerate(self.seg_classes.values()):
            y_mask[i, labels] = 1

        return data, slices, y_mask

    @property
    def raw_file_names(self):
        return list(self.category_ids.values()) + ["train_test_split"]

    @property
    def processed_raw_paths(self):
        cats = "_".join([cat[:3].lower() for cat in self.categories])
        processed_raw_paths = [os.path.join(self.processed_dir, "raw_{}_{}".format(
            cats, s)) for s in ["train", "val", "test", "trainval"]]
        return processed_raw_paths

    @property
    def processed_file_names(self):
        cats = "_".join([cat[:3].lower() for cat in self.categories])
        return [os.path.join("{}_{}.pt".format(cats, split)) for split in ["train", "val", "test", "trainval"]]

    def download(self):
        if self.is_test:
            return
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        name = self.url.split("/")[-1].split(".")[0]
        os.rename(osp.join(self.root, name), self.raw_dir)

    def get_raw_data(self, idx, **kwargs):
        data = self.raw_data.__class__()

        if hasattr(self.raw_data, '__num_nodes__'):
            data.num_nodes = self.raw_data.__num_nodes__[idx]

        for key in self.raw_data.keys:
            item, slices = self.raw_data[key], self.raw_slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()
            # print(slices[idx], slices[idx + 1])
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.raw_data.__cat_dim__(key, item)] = slice(start, end)
            elif start + 1 == end:
                s = slices[start]
            else:
                s = slice(start, end)
            data[key] = item[s]
        return data

    def _process_filenames(self, filenames):
        data_raw_list = []
        data_list = []
        categories_ids = [self.category_ids[cat] for cat in self.categories]
        cat_idx = {categories_ids[i]: i for i in range(len(categories_ids))}

        has_pre_transform = self.pre_transform is not None

        id_scan = -1
        for name in tq(filenames):
            cat = name.split(osp.sep)[0]
            if cat not in categories_ids:
                continue
            id_scan += 1
            data = read_txt_array(osp.join(self.raw_dir, name))
            pos = data[:, :3]
            x = data[:, 3:6]
            y = data[:, -1].type(torch.long)
            category = torch.ones(x.shape[0], dtype=torch.long) * cat_idx[cat]
            id_scan_tensor = torch.from_numpy(np.asarray([id_scan])).clone()
            data = Data(pos=pos, x=x, y=y, category=category,
                        id_scan=id_scan_tensor)
            data = SaveOriginalPosId()(data)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            data_raw_list.append(data.clone() if has_pre_transform else data)
            if has_pre_transform:
                data = self.pre_transform(data)
                data_list.append(data)
        if not has_pre_transform:
            return [], data_raw_list
        return data_raw_list, data_list

    def _save_data_list(self, datas, path_to_datas, save_bool=True):
        if save_bool:
            torch.save(self.collate(datas), path_to_datas)

    def _re_index_trainval(self, trainval):
        if len(trainval) == 0:
            return trainval
        train, val = trainval
        for v in val:
            v.id_scan += len(train)
        assert (train[-1].id_scan + 1 ==
                val[0].id_scan).item(), (train[-1].id_scan, val[0].id_scan)
        return train + val

    def process(self):
        if self.is_test:
            return
        raw_trainval = []
        trainval = []
        for i, split in enumerate(["train", "val", "test"]):
            path = osp.join(self.raw_dir, "train_test_split",
                            f"shuffled_{split}_file_list.json")
            with open(path, "r") as f:
                filenames = [
                    osp.sep.join(name.split('/')[1:]) + ".txt" for name in json.load(f)
                ]  # Removing first directory.
            data_raw_list, data_list = self._process_filenames(
                sorted(filenames))
            if split == "train" or split == "val":
                if len(data_raw_list) > 0:
                    raw_trainval.append(data_raw_list)
                trainval.append(data_list)

            self._save_data_list(data_list, self.processed_paths[i])
            self._save_data_list(
                data_raw_list, self.processed_raw_paths[i], save_bool=len(data_raw_list) > 0)

        self._save_data_list(self._re_index_trainval(
            trainval), self.processed_paths[3])
        self._save_data_list(self._re_index_trainval(
            raw_trainval), self.processed_raw_paths[3], save_bool=len(raw_trainval) > 0)

    def __repr__(self):
        return "{}({}, categories={})".format(self.__class__.__name__, len(self), self.categories)


class ShapeNetDataset(BaseDataset):
    """ Wrapper around ShapeNet that creates train and test datasets.

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - category: List of categories or All
            - normal: bool, include normals or not
            - pre_transforms
            - train_transforms
            - test_transforms
            - val_transforms
    """

    FORWARD_CLASS = "forward.shapenet.ForwardShapenetDataset"

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        try:
            self._category = dataset_opt.category
            is_test = dataset_opt.get("is_test", False)
        except KeyError:
            self._category = None

        self.train_dataset = ShapeNet(
            self._data_path,
            self._category,
            include_normals=dataset_opt.normal,
            split="train",
            pre_transform=self.pre_transform,
            transform=self.train_transform,
            is_test=is_test,
        )

        self.val_dataset = ShapeNet(
            self._data_path,
            self._category,
            include_normals=dataset_opt.normal,
            split="val",
            pre_transform=self.pre_transform,
            transform=self.val_transform,
            is_test=is_test,
        )

        self.test_dataset = ShapeNet(
            self._data_path,
            self._category,
            include_normals=dataset_opt.normal,
            split="test",
            transform=self.test_transform,
            pre_transform=self.pre_transform,
            is_test=is_test,
        )
        self._categories = self.train_dataset.categories

    @property  # type: ignore
    @save_used_properties
    def class_to_segments(self):
        classes_to_segment = {}
        for key in self._categories:
            classes_to_segment[key] = ShapeNet.seg_classes[key]
        return classes_to_segment

    @property
    def is_hierarchical(self):
        return len(self._categories) > 1

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return ShapenetPartTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
