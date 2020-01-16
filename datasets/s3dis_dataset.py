import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
import h5py
import torch
import glob
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import logging

from .base_dataset import BaseDataset
import datasets.transforms as cT
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm as tq
import csv

log = logging.getLogger(__name__)

def object_name_to_label(object_class):
    """convert from object name in S3DIS to an int"""
    object_label = {
        'ceiling': 1,
        'floor': 2,
        'wall': 3,
        'column': 4,
        'beam': 5,
        'window': 6,
        'door': 7,
        'table': 8,
        'chair': 9,
        'bookcase': 10,
        'sofa': 11,
        'board': 12,
        'clutter': 13,
        'stairs': 0,
    }.get(object_class, 0)
    return object_label


def read_s3dis_format(train_file, room_name, label_out=True, verbose=False, debug=False):
    """extract data from a room folder"""
    raw_path = osp.join(train_file, '{}.txt'.format(room_name))
    if debug:
        reader = pd.read_csv(raw_path, delimiter='\n')
        RECOMMENDED = 6
        for idx, row in enumerate(reader.values):
            row = row[0].split(' ')
            if (len(row) != RECOMMENDED):
                print("1: {} row {}: {}".format(raw_path, idx, row))

            try:
                for r in row:
                    r = float(r)
            except:
                print("2: {} row {}: {}".format(raw_path, idx, row))

        return True
    else:
        room_ver = pd.read_csv(raw_path, sep=' ', header=None).values
        xyz = np.ascontiguousarray(room_ver[:, 0:3], dtype='float32')
        try:
            rgb = np.ascontiguousarray(room_ver[:, 3:6], dtype='uint8')
        except ValueError:
            rgb = np.zeros((room_ver.shape[0], 3), dtype='uint8')
            print('WARN - corrupted rgb data for file %s' % raw_path)
        if not label_out:
            return xyz, rgb
        n_ver = len(room_ver)
        del room_ver
        nn = NearestNeighbors(1, algorithm='kd_tree').fit(xyz)
        room_labels = np.zeros((n_ver,), dtype='int64')
        room_object_indices = np.zeros((n_ver,), dtype='int64')
        objects = glob.glob(osp.join(train_file, "Annotations/*.txt"))
        i_object = 1
        for single_object in objects:
            object_name = os.path.splitext(os.path.basename(single_object))[0]
            if verbose:
                print("adding object " + str(i_object) + " : " + object_name)
            object_class = object_name.split('_')[0]
            object_label = object_name_to_label(object_class)
            obj_ver = pd.read_csv(single_object, sep=' ', header=None).values
            _, obj_ind = nn.kneighbors(obj_ver[:, 0:3])
            room_labels[obj_ind] = object_label
            room_object_indices[obj_ind] = i_object
            i_object = i_object + 1

        return torch.from_numpy(xyz), torch.from_numpy(rgb), \
            torch.from_numpy(room_labels), torch.from_numpy(room_object_indices)


class S3DIS(InMemoryDataset):
    r"""The (pre-processed) Stanford Large-Scale 3D Indoor Spaces dataset from
    the `"3D Semantic Parsing of Large-Scale Indoor Spaces"
    <http://buildingparser.stanford.edu/images/3D_Semantic_Parsing.pdf>`_
    paper, containing point clouds of six large-scale indoor parts in three
    buildings with 12 semantic elements (and one clutter class).

    Args:
        root (string): Root directory where the dataset should be saved.
        test_area (int, optional): Which area to use for testing (1-6).
            (default: :obj:`6`)
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

    url = ("https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1")
    zip_name = "Stanford3dDataset_v1.2_Aligned_Version.zip"
    folders = ["Area_{}".format(i) for i in range(1, 7)]

    def __init__(self,
                 root,
                 test_area=6,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_collate_transform=None,
                 pre_filter=None,
                 keep_instance=False,
                 verbose=False,
                 debug=False):
        assert test_area >= 1 and test_area <= 6
        self.pre_collate_transform = pre_collate_transform
        self.test_area = test_area
        self.keep_instance = keep_instance
        self.verbose = verbose
        self.debug = debug
        super(S3DIS, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return self.folders

    @property
    def processed_file_names(self):
        test_area = self.test_area
        return ['{}_{}.pt'.format(s, test_area) for s in ['train', 'test']]

    def download(self):
        raw_folders = os.listdir(self.raw_dir)
        if len(raw_folders) == 0:
            raise RuntimeError(
                'Dataset not found. Please download {} from {} and move it to {} with {}'.
                format(self.zip_name, self.url, self.raw_dir, self.folders))
        else:
            intersection = len(set(self.folders).intersection(set(raw_folders)))
            if intersection == 0:
                print('The data seems properly downloaded')
            else:
                raise RuntimeError(
                    'Dataset not found. Please download {} from {} and move it to {} with {}'.
                    format(self.zip_name, self.url, self.raw_dir, self.folders))

    def process(self):

        train_areas = [f for f in self.folders if str(self.test_area) not in f]
        test_areas = [f for f in self.folders if str(self.test_area) in f]

        train_files = [(f, room_name, osp.join(self.raw_dir, f, room_name)) for f in train_areas
                       for room_name in os.listdir(osp.join(self.raw_dir, f)) if '.DS_Store' != room_name]

        test_files = [(f, room_name, osp.join(self.raw_dir, f, room_name)) for f in test_areas
                      for room_name in os.listdir(osp.join(self.raw_dir, f)) if '.DS_Store' != room_name]

        train_data_list, test_data_list = [], []

        for (area, room_name, file_path) in tq(train_files + test_files):

            if self.debug:
                read_s3dis_format(
                    file_path, room_name, label_out=True, verbose=self.verbose, debug=self.debug)
            else:
                xyz, rgb, room_labels, room_object_indices = read_s3dis_format(
                    file_path, room_name, label_out=True, verbose=self.verbose, debug=self.debug)

                data = Data(pos=xyz, x=rgb.float(), y=room_labels)

                if self.keep_instance:
                    data.room_object_indices = room_object_indices

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                if (area, room_name, file_path) in train_files:
                    train_data_list.append(data)
                else:
                    test_data_list.append(data)

        if self.pre_collate_transform:
            train_data_list = self.pre_collate_transform.fit_transform(train_data_list)
            test_data_list = self.pre_collate_transform.transform(test_data_list)

        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(test_data_list), self.processed_paths[1])

"""
class NormalizeMeanStd(object):

    def __init__(self, keys):
        self._keys = keys

    def fit(self, data_list):
        for key in self._keys:
            if key in data_list[0]:
                arr = [getattr(d, key) for d in data_list]
                arr = torch.cat(arr, dim=0)
                setattr(self, "{}_mean".format(key), torch.mean(arr, dim=-1))
                setattr(self, "{}_std".format(key), torch.std(arr, dim=-1))

    def transform(self, data_list):
        for key in self._keys:
            if key in data_list[0]:
                data_list = [setattr((getattr(d, key) - getattr(self, '{}_mean'))/(getattr(d, '{}_std')))
                             for d in data_list]

    def fit_transform(self, data_list):
        self.fit(data_list)
        return self.transform(data_list)
"""


class S3DIS_With_Weights(S3DIS):
    def __init__(
        self,
        root,
        test_area=6,
        train=True,
        transform=None,
        pre_transform=None,
        pre_collate_transform=None,
        pre_filter=None,
        class_weight_method=None,
    ):
        super(S3DIS_With_Weights, self).__init__(
            root,
            test_area=test_area,
            train=train,
            transform=transform,
            pre_transform=pre_transform,
            pre_collate_transform=pre_collate_transform,
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
                self.idx_classes, weights = torch.unique(self.data.y, return_counts=True)
                weights = weights.float()
                weights = weights.mean() / weights
                if class_weight_method == "sqrt":
                    weights = torch.sqrt(weights)
                elif str(class_weight_method).startswith("log"):
                    w = float(class_weight_method.replace("log", ""))
                    weights = 1 / torch.log(1.1 + weights / weights.sum())

                weights /= torch.sum(weights)
            log.info(
                "CLASS WEIGHT : {}".format(
                    {name: np.round(weights[index].item(), 4) for index, name in inv_class_map.items()}
                )
            )
            self.weight_classes = weights
        else:
            self.weight_classes = torch.ones((len(inv_class_map.keys())))


class S3DISDataset(BaseDataset):
    def __init__(self, dataset_opt, training_opt):
        super().__init__(dataset_opt, training_opt)
        self._data_path = os.path.join(dataset_opt.dataroot, "S3DIS")


        pre_transform = cT.GridSampling(dataset_opt.first_subsampling, 13)
        # Select only 2^15 points from the room
        # pre_transform = T.FixedPoints(dataset_opt.room_points)

        transform = T.Compose(
            [T.FixedPoints(dataset_opt.num_points), T.RandomTranslate(0.01), T.RandomRotate(180, axis=2),]
        )

        train_dataset = S3DIS_With_Weights(
            self._data_path,
            test_area=self.dataset_opt.fold,
            train=True,
            pre_transform=pre_transform,
            transform=transform,
            class_weight_method=dataset_opt.class_weight_method,
        )
        test_dataset = S3DIS_With_Weights(
            self._data_path,
            test_area=self.dataset_opt.fold,
            train=False,
            pre_transform=pre_transform,
            transform=T.FixedPoints(dataset_opt.num_points),
        )

        self._create_dataloaders(train_dataset, test_dataset, validation=None)
