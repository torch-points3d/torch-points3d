import numpy as np
import os
import os.path as osp
import torch
from torch_geometric.data import Data

from torch_points3d.datasets.registration.base_kitti import BaseKitti
from torch_points3d.datasets.registration.utils import PatchExtractor


# from torch_points3d.metrics.registration_tracker import PatchRegistrationTracker
from torch_points3d.metrics.registration_tracker import FragmentRegistrationTracker

from torch_points3d.datasets.registration.base_siamese_dataset import BaseSiameseDataset
from torch_points3d.datasets.registration.base_siamese_dataset import GeneralFragment


class FragmentKitti(BaseKitti, GeneralFragment):
    """
    Fragment from KITTI Odometry dataset
    """

    def __init__(self, root,
                 mode='train',
                 self_supervised=False,
                 min_size_block=0.3,
                 max_size_block=2,
                 max_dist_overlap=0.01,
                 max_time_distance=3,
                 min_dist=10,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 is_online_matching=False,
                 num_pos_pairs=1024,
                 ss_transform=None,
                 min_points=300):
        BaseKitti.__init__(self,
                           root,
                           mode,
                           max_dist_overlap,
                           max_time_distance,
                           min_dist,
                           transform,
                           pre_transform,
                           pre_filter)

        self.path_match = osp.join(self.processed_dir, self.mode, "matches")
        self.list_fragment = [f for f in os.listdir(self.path_match) if "matches" in f]
        self.is_online_matching = is_online_matching
        self.num_pos_pairs = num_pos_pairs
        self.self_supervised = self_supervised
        self.min_size_block = min_size_block
        self.max_size_block = max_size_block
        self.ss_transform = ss_transform
        self.min_points = min_points

    def get(self, idx):
        return self.get_fragment(idx)

    def __len__(self):
        return len(self.list_fragment)

    def len(self):
        return len(self)

    def process(self):
        super().process()

    def download(self):
        super().download()


class KittiDataset(BaseSiameseDataset):

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        pre_transform = self.pre_transform
        self.ss_transform = getattr(self, "ss_transform", None)
        train_transform = self.train_transform
        test_transform = self.test_transform
        pre_filter = self.pre_filter

        self.train_dataset = FragmentKitti(
            root=self._data_path,
            mode="train",
            self_supervised=dataset_opt.self_supervised,
            min_size_block=dataset_opt.min_size_block,
            max_size_block=dataset_opt.max_size_block,
            max_dist_overlap=dataset_opt.max_dist_overlap,
            max_time_distance=dataset_opt.max_time_distance,
            min_dist=dataset_opt.min_dist,
            ss_transform=self.ss_transform,
            pre_transform=pre_transform,
            transform=train_transform,
            pre_filter=pre_filter,
            is_online_matching=dataset_opt.is_online_matching,
            num_pos_pairs=dataset_opt.num_pos_pairs,
            min_points=dataset_opt.min_points)

        self.val_dataset = FragmentKitti(
            root=self._data_path,
            mode="val",
            self_supervised=False,
            max_dist_overlap=dataset_opt.max_dist_overlap,
            max_time_distance=dataset_opt.max_time_distance,
            min_dist=dataset_opt.min_dist,
            pre_transform=pre_transform,
            transform=test_transform,
            pre_filter=pre_filter,
            is_online_matching=dataset_opt.is_online_matching,
            num_pos_pairs=dataset_opt.num_pos_pairs)

        self.test_dataset = FragmentKitti(
            root=self._data_path,
            mode="test",
            self_supervised=False,
            max_dist_overlap=dataset_opt.max_dist_overlap,
            max_time_distance=dataset_opt.max_time_distance,
            min_dist=dataset_opt.min_dist,
            pre_transform=pre_transform,
            transform=test_transform,
            pre_filter=pre_filter,
            is_online_matching=dataset_opt.is_online_matching,
            num_pos_pairs=dataset_opt.num_pos_pairs)
