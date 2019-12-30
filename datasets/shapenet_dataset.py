import os
import json

import torch
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from torch_geometric.io import read_txt_array

from .base_dataset import BaseDataset


class ShapeNet(InMemoryDataset):
    r""" The ShapeNet part level segmentation dataset from the `"A Scalable
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
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
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

    url_with_normal = "https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"

    category_ids = {
        'Airplane': '02691156',
        'Bag': '02773838',
        'Cap': '02954340',
        'Car': '02958343',
        'Chair': '03001627',
        'Earphone': '03261776',
        'Guitar': '03467517',
        'Knife': '03624134',
        'Lamp': '03636649',
        'Laptop': '03642806',
        'Motorbike': '03790512',
        'Mug': '03797390',
        'Pistol': '03948459',
        'Rocket': '04099429',
        'Skateboard': '04225987',
        'Table': '04379243',
    }

    def __init__(self, root, categories=None, normal=True, train=True, transform=None,
                 pre_transform=None, pre_filter=None):
        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)
        self.categories = categories
        self.normal = normal
        if not self.normal:
            if transform:
                transform = T.Compose([transform, NoNormalTransform()])
            else:
                transform = NoNormalTransform()
        super(ShapeNet, self).__init__(root, transform, pre_transform,
                                       pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return self.category_ids.values()

    @property
    def processed_file_names(self):
        cats = '_'.join([cat[:3].lower() for cat in self.categories])
        return [
            os.path.join('{}_{}.pt'.format(cats, s)) for s in ['training', 'test']
        ]

    def download(self):
        path = download_url(self.url_with_normal, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process_raw_files(self, raw_file_paths):
        data_list = []
        categories_ids_to_process = [self.category_ids[cat] for cat in self.categories]
        cat_idx = {self.category_ids[cat]: i for i, cat in enumerate(self.categories)}
        for raw_file in raw_file_paths:
            cat = raw_file.split(os.path.sep)[0]
            if cat not in categories_ids_to_process:
                continue

            data = read_txt_array(os.path.join(self.raw_dir, raw_file))
            if self.normal:
                x = data[:, 3:6]
            else:
                x = None
            data = Data(y=data[:, -1].type(torch.long), pos=data[:, 0:3], x=x, category=cat_idx[cat])
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list

    def process(self):
        file_split = {}
        for split in ['train', 'val', 'test']:
            with open(os.path.join(self.raw_dir, 'train_test_split', 'shuffled_%s_file_list.json' % split), 'r') as fp:
                raw_files = json.load(fp)
                raw_files = [os.path.sep.join(f.split(os.path.sep)[1:]) + '.txt'
                             for f in raw_files]  # removing first directory, useless here
                file_split[split] = raw_files

        train_data_list = self.process_raw_files(file_split['train'])
        val_data_list = self.process_raw_files(file_split['val'])
        test_data_list = self.process_raw_files(file_split['test'])

        data = self.collate(train_data_list + val_data_list)
        torch.save(data, self.processed_paths[0])
        torch.save(self.collate(test_data_list), self.processed_paths[1])

    def __repr__(self):
        return '{}({}, categories={}, normal: {})'.format(self.__class__.__name__,
                                                          len(self), self.categories, self.normal)


class NoNormalTransform:
    def __call__(self, data):
        data.x = None
        return data


class ShapeNetDataset(BaseDataset):
    def __init__(self, dataset_opt, training_opt):
        super().__init__(dataset_opt, training_opt)
        self._data_path = os.path.join(dataset_opt.dataroot, 'ShapeNet')
        self._category = dataset_opt.category
        pre_transform = T.NormalizeScale()
        train_dataset = ShapeNet(self._data_path, self._category, normal=dataset_opt.normal, train=True,
                                 pre_transform=pre_transform)
        test_dataset = ShapeNet(self._data_path, self._category,  normal=dataset_opt.normal, train=False,
                                pre_transform=pre_transform)

        self._create_dataloaders(train_dataset, test_dataset, validation=None)
