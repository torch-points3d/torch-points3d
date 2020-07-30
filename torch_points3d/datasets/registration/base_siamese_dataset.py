import numpy as np
import os.path as osp
import random
import torch
from torch_geometric.data import Data
from torch_points_kernels.points_cpu import ball_query

from torch_points3d.core.data_transform import MultiScaleTransform
from torch_points3d.core.data_transform import PairTransform
from torch_points3d.datasets.registration.pair import DensePairBatch
from torch_points3d.utils.enums import ConvolutionFormat
from torch_points3d.utils.config import ConvolutionFormatFactory
from torch_points3d.datasets.registration.pair import PairMultiScaleBatch, PairBatch
from torch_points3d.datasets.registration.pair import Pair, MultiScalePair
from torch_points3d.datasets.registration.utils import tracked_matches
from torch_points3d.datasets.registration.utils import compute_overlap_and_matches
from torch_points3d.datasets.base_dataset import BaseDataset


class BaseSiameseDataset(BaseDataset):
    def __init__(self, dataset_opt):
        """
        base dataset for siamese inputs
        """
        super().__init__(dataset_opt)

    @staticmethod
    def _get_collate_function(conv_type, is_multiscale):

        is_dense = ConvolutionFormatFactory.check_is_dense_format(conv_type)

        if is_multiscale:
            if conv_type.lower() == ConvolutionFormat.PARTIAL_DENSE.value.lower():
                return lambda datalist: PairMultiScaleBatch.from_data_list(datalist)
            else:
                raise NotImplementedError(
                    "MultiscaleTransform is activated and supported only for partial_dense format"
                )

        if is_dense:
            return lambda datalist: DensePairBatch.from_data_list(datalist)
        else:
            return lambda datalist: PairBatch.from_data_list(datalist)


class GeneralFragment(object):

    """
    implementation of get_fragment and get_name to avoid repetitions
    """

    def get_raw_pair(self, idx):
        """
        get the pair before the data augmentation
        """
        match = np.load(osp.join(self.path_match, "matches{:06d}.npy".format(idx)),
                        allow_pickle=True).item()
        if(not self.self_supervised):
            data_source = torch.load(match["path_source"]).to(torch.float)
            data_target = torch.load(match["path_target"]).to(torch.float)
            new_pair = torch.from_numpy(match["pair"])
        else:
            if(random.random() < 0.5):
                data_source_o = torch.load(match["path_source"]).to(torch.float)
                data_target_o = torch.load(match["path_source"]).to(torch.float)
            else:
                data_source_o = torch.load(match["path_target"]).to(torch.float)
                data_target_o = torch.load(match["path_target"]).to(torch.float)
            data_source, data_target, new_pair = self.unsupervised_preprocess(
                data_source_o, data_target_o)

        return data_source, data_target, new_pair

    def unsupervised_preprocess(self, data_source_o, data_target_o):
        """
        same pairs for self supervised learning
        """
        len_col = 0

        while(len_col < self.min_points):
            # choose only one data augmentation randomly in the ss_transform (usually a crop)
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
        return data_source, data_target, new_pair

    def get_fragment(self, idx):

        data_source, data_target, new_pair = self.get_raw_pair(idx)
        if self.transform is not None:
            data_source = self.transform(data_source)
            data_target = self.transform(data_target)

        if(hasattr(data_source, "multiscale")):
            batch = MultiScalePair.make_pair(data_source, data_target)
        else:
            batch = Pair.make_pair(data_source, data_target)
        if self.is_online_matching:
            new_match = compute_overlap_and_matches(
                Data(pos=data_source.pos), Data(pos=data_target.pos), self.max_dist_overlap
            )
            batch.pair_ind = torch.from_numpy(new_match["pair"].copy())
        else:
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

    def get_name(self, idx):
        """
        get the name of the scene and the name of the fragments.
        """

        match = np.load(osp.join(self.path_match,
                                 "matches{:06d}.npy".format(idx)),
                        allow_pickle=True).item()
        source = match["name_source"]
        target = match["name_target"]
        scene = match["scene"]
        return scene, source, target
