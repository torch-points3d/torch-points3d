import numpy as np
import random
from itertools import repeat, product
import os
import os.path as osp
import torch

from torch_geometric.data import Data
from torch_points_kernels.points_cpu import ball_query

from torch_points3d.datasets.classification.modelnet import SampledModelNet
from torch_points3d.datasets.registration.base_siamese_dataset import BaseSiameseDataset
from torch_points3d.datasets.registration.base_siamese_dataset import GeneralFragment
from torch_points3d.metrics.registration_tracker import FragmentRegistrationTracker
from torch_points3d.datasets.registration.pair import Pair, MultiScalePair
from torch_points3d.datasets.registration.utils import tracked_matches
from torch_points3d.datasets.registration.utils import compute_overlap_and_matches


class SiameseModelNet(SampledModelNet, GeneralFragment):
    r"""
    the ModelNet Dataset from the `"3D ShapeNets: A Deep
    Representation for Volumetric Shapes"
    <https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf>`_ paper,
    containing sampled CAD models of 40 categories. Each sample contains 10,000
    points uniformly sampled with their normal vector.

    But applied for registration.
    Only the self supervised mode is supported
    """

    def __init__(self, root,
                 name_modelnet="10",
                 min_size_block=0.3,
                 max_size_block=2,
                 max_dist_overlap=0.1,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 num_pos_pairs=1024,
                 ss_transform=None,
                 min_points=500,
    ):
        SampledModelNet.__init__(self,
                                 root,
                                 name_modelnet,
                                 train,
                                 transform,
                                 pre_transform,
                                 pre_filter)
        self.self_supervised = True
        self.num_pos_pairs = num_pos_pairs
        self.min_size_block = min_size_block
        self.max_size_block = max_size_block
        self.max_dist_overlap = max_dist_overlap
        self.ss_transform = ss_transform
        self.min_points = min_points
        self.train = train
        if(self.train):
            self.name = "train"
        else:
            self.name = "test"

    def get_model(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()
            # print(slices[idx], slices[idx + 1])
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(start, end)
            elif start + 1 == end:
                s = slices[start]
            else:
                s = slice(start, end)
            data[key] = item[s]
        return data

    def get_fragment(self, idx):

        data_source_o = self.get_model(idx)
        data_target_o = self.get_model(idx)

        len_col = 0

        while(len_col < self.min_points):
            if(self.ss_transform is not None):
                n1 = np.random.randint(0, len(self.ss_transform.transforms))
                t1 = self.ss_transform.transforms[n1]
                n2 = np.random.randint(0, len(self.ss_transform.transforms))
                t2 = self.ss_transform.transforms[n2]
                data_source = t1(data_source_o.clone())
                data_target = t2(data_target_o.clone())
            else:
                data_source = data_source_o
                data_target = data_target_o
            pos = data_source.pos
            i = torch.randint(0, len(pos), (1,))
            size_block = random.random()*(self.max_size_block - self.min_size_block) + self.min_size_block
            point = pos[i].view(1, 3)
            ind, dist = ball_query(point,
                                   pos,
                                   radius=size_block,
                                   max_num=-1,
                                   mode=1)
            _, col = ind[dist[:, 0] > 0].t()

            ind_t, dist_t = ball_query(data_target.pos,
                                       pos[col],
                                       radius=self.max_dist_overlap,
                                       max_num=1,
                                       mode=1)
            col_target, ind_col = ind_t[dist_t[:, 0] > 0].t()
            col = col[ind_col]

            new_pair = torch.stack((col, col_target)).T
            len_col = len(new_pair)

        if self.transform is not None:
            data_source = self.transform(data_source)
            data_target = self.transform(data_target)

        if(hasattr(data_source, "multiscale")):
            batch = MultiScalePair.make_pair(data_source, data_target)
        else:
            batch = Pair.make_pair(data_source, data_target)

        pair = tracked_matches(data_source, data_target, new_pair)
        batch.pair_ind = pair

        num_pos_pairs = len(batch.pair_ind)
        if self.num_pos_pairs < len(batch.pair_ind):
            num_pos_pairs = self.num_pos_pairs

        rand_ind = torch.randperm(len(batch.pair_ind))[:num_pos_pairs]
        batch.pair_ind = batch.pair_ind[rand_ind]
        batch.size_pair_ind = torch.tensor([num_pos_pairs])
        if(len(batch.pair_ind) == 0):
            print("Warning")
        return batch.contiguous()

    def __getitem__(self, idx):
        res = self.get_fragment(idx)
        return res

    def get_name(self, idx):
        data = self.get_model(idx)
        return data.y.item(), "{}_source".format(idx), "{}_target".format(idx)


class SiameseModelNetDataset(BaseSiameseDataset):

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        pre_transform = self.pre_transform
        ss_transform = getattr(self, "ss_transform", None)
        train_transform = self.train_transform
        test_transform = self.test_transform
        pre_filter = self.pre_filter
        self.tau_1 = dataset_opt.tau_1
        self.rot_thresh = dataset_opt.rot_thresh
        self.trans_thresh = dataset_opt.trans_thresh

        self.train_dataset = SiameseModelNet(
            root=self._data_path,
            name_modelnet=dataset_opt.name_modelnet,
            train=True,
            min_size_block=dataset_opt.min_size_block,
            max_size_block=dataset_opt.max_size_block,
            max_dist_overlap=dataset_opt.max_dist_overlap,
            pre_transform=pre_transform,
            transform=train_transform,
            pre_filter=pre_filter,
            num_pos_pairs=dataset_opt.num_pos_pairs,
            ss_transform=ss_transform,
            min_points=dataset_opt.min_points)

        self.test_dataset = SiameseModelNet(
            root=self._data_path,
            name_modelnet=dataset_opt.name_modelnet,
            train=False,
            min_size_block=dataset_opt.min_size_block,
            max_size_block=dataset_opt.max_size_block,
            max_dist_overlap=dataset_opt.max_dist_overlap,
            pre_transform=pre_transform,
            transform=test_transform,
            pre_filter=pre_filter,
            num_pos_pairs=dataset_opt.num_pos_pairs,
            min_points=dataset_opt.min_points)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """
        Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return FragmentRegistrationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log, tau_1=self.tau_1, rot_thresh=self.rot_thresh, trans_thresh=self.trans_thresh)
