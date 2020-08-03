import numpy as np
import os
import os.path as osp
import torch
from torch_geometric.data import Data

from torch_points3d.datasets.registration.base3dmatch import Base3DMatch
from torch_points3d.datasets.registration.utils import PatchExtractor
from torch_points3d.datasets.registration.utils import tracked_matches
from torch_points3d.datasets.registration.pair import Pair, MultiScalePair
from torch_points3d.metrics.registration_tracker import PatchRegistrationTracker
from torch_points3d.metrics.registration_tracker import FragmentRegistrationTracker

from torch_points3d.datasets.registration.base_siamese_dataset import BaseSiameseDataset
from torch_points3d.datasets.registration.base_siamese_dataset import GeneralFragment
from torch_points3d.datasets.registration.utils import compute_overlap_and_matches
from torch_points3d.datasets.registration.test3dmatch import TestPair3DMatch



class Patch3DMatch(Base3DMatch):
    def __init__(
        self,
        root,
        radius_patch=0.3,
        num_frame_per_fragment=50,
        mode="train_small",
        min_overlap_ratio=0.3,
        max_overlap_ratio=1.0,
        max_dist_overlap=0.01,
        tsdf_voxel_size=0.02,
        limit_size=700,
        depth_thresh=6,
        is_fine=True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        verbose=False,
        debug=False,
        num_random_pt=5000,
        is_offline=False,
        pre_transform_patch=None,
    ):
        r"""
        Patch extracted from :the Princeton 3DMatch dataset\n
        `"3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions"
        <https://arxiv.org/pdf/1603.08182.pdf>`_
        paper, containing rgbd frames of the following dataset:
        `" SUN3D: A Database of Big Spaces Reconstructed using SfM and Object Labels
        "<http://sun3d.cs.princeton.edu/>`
        `"Scene Coordinate Regression Forests for Camera Relocalization in RGB-D Images
        "<https://www.microsoft.com/en-us/research/publication/scene-coordinate-regression-forests-for-camera-relocalization-in-rgb-d-images/>`
        `"Unsupervised Feature Learning for 3D Scene Labeling
        "<http://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/>`
        `"BundleFusion: Real-time Globally Consistent 3D Reconstruction using Online
        Surface Re-integration
        "<http://graphics.stanford.edu/projects/bundlefusion/>`
        `"Learning to Navigate the Energy Landscape
        "<http://graphics.stanford.edu/projects/reloc/>`

        Args:

            root (string): Root directory where the dataset should be saved

            radius_patch(float, optional): the size of the patch

            num_frame_per_fragment (int, optional): indicate the number of frames
                we use to build fragments. If it is equal to 0, then we don't
                build fragments and use the raw frames.

            mode (string, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)

            min_overlap_ratio(float, optional): the minimum overlap we should have to match two fragments (overlap is the number of points in a fragment that matches in an other fragment divided by the number of points)

            max_dist_overlap(float, optional): minimum distance to consider that a point match with an other.
            tsdf_voxel_size(float, optional): the size of the tsdf voxel grid to perform fine RGBD fusion to create fine fragments
            depth_thresh: threshold to remove depth pixel that are two far.

            is_fine: fine mode for the fragment fusion

            limit_size : limit the number of pixel at each direction to abvoid to heavy tsdf voxel

            transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                every access. (default: :obj:`None`)

            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)
            num_random_pt: number of point we select
        """
        self.is_patch = True
        super(Patch3DMatch, self).__init__(
            root,
            num_frame_per_fragment,
            mode,
            min_overlap_ratio,
            max_overlap_ratio,
            max_dist_overlap,
            tsdf_voxel_size,
            limit_size,
            depth_thresh,
            is_fine,
            transform,
            pre_transform,
            pre_filter,
            verbose,
            debug,
            num_random_pt,
            is_offline,
            radius_patch,
            pre_transform_patch,
        )

        self.radius_patch = radius_patch
        self.is_offline = is_offline
        self.path_data = osp.join(self.processed_dir, self.mode, "matches")
        if self.is_offline:
            self.path_data = osp.join(self.processed_dir, self.mode, "patches")

    def get_patch_online(self, idx):
        p_extractor = PatchExtractor(self.radius_patch)

        match = np.load(osp.join(self.path_data, "matches{:06d}.npy".format(idx)), allow_pickle=True).item()
        data_source = torch.load(match["path_source"]).to(torch.float)
        data_target = torch.load(match["path_target"]).to(torch.float)

        # select a random match on the list of match.
        # It cannot be 0 because matches are filtered.
        rand = np.random.randint(0, len(match["pair"]))

        data_source = p_extractor(data_source, match["pair"][rand][0])
        data_target = p_extractor(data_target, match["pair"][rand][1])

        if self.transform is not None:
            data_source = self.transform(data_source)
            data_target = self.transform(data_target)
        batch = Pair.make_pair(data_source, data_target)
        batch = batch.contiguous()
        return batch

    def get_patch_offline(self, idx):
        data_source = torch.load(osp.join(self.path_data, "patches_source{:06d}.pt".format(idx)))
        data_target = torch.load(osp.join(self.path_data, "patches_target{:06d}.pt".format(idx)))
        if self.transform is not None:
            data_source = self.transform(data_source)
            data_target = self.transform(data_target)

        if(hasattr(data_source, "multiscale")):
            batch = MultiScalePair.make_pair(data_source, data_target)
        else:
            batch = Pair.make_pair(data_source, data_target)
        return batch.contiguous()

    def get(self, idx):
        if self.is_offline:
            return self.get_patch_offline(idx)
        else:
            return self.get_patch_online(idx)

    def __len__(self):
        size_dataset = len(os.listdir(self.path_data))
        if self.is_offline:
            size_dataset = size_dataset // 2
        return size_dataset


class Fragment3DMatch(Base3DMatch, GeneralFragment):
    r"""
        Fragment extracted from :the Princeton 3DMatch dataset\n
        `"3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions"
        <https://arxiv.org/pdf/1603.08182.pdf>`_
        paper, containing rgbd frames of the following dataset:
        `" SUN3D: A Database of Big Spaces Reconstructed using SfM and Object Labels
        "<http://sun3d.cs.princeton.edu/>`
        `"Scene Coordinate Regression Forests for Camera Relocalization in RGB-D Images
        "<https://www.microsoft.com/en-us/research/publication/scene-coordinate-regression-forests-for-camera-relocalization-in-rgb-d-images/>`
        `"Unsupervised Feature Learning for 3D Scene Labeling
        "<http://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/>`
        `"BundleFusion: Real-time Globally Consistent 3D Reconstruction using Online
        Surface Re-integration
        "<http://graphics.stanford.edu/projects/bundlefusion/>`
        `"Learning to Navigate the Energy Landscape
        "<http://graphics.stanford.edu/projects/reloc/>`

        Args:

            root (string): Root directory where the dataset should be saved

            num_frame_per_fragment (int, optional): indicate the number of frames
                we use to build fragments. If it is equal to 0, then we don't
                build fragments and use the raw frames.

            mode (string, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)

            min_overlap_ratio(float, optional): the minimum overlap we should have to match two fragments (overlap is the number of points in a fragment that matches in an other fragment divided by the number of points)
            max_overlap_ratio(float, optional): the maximum overlap we should have to match two fragments
            max_dist_overlap(float, optional): minimum distance to consider that a point match with an other.
            tsdf_voxel_size(float, optional): the size of the tsdf voxel grid to perform fine RGBD fusion to create fine fragments
            depth_thresh: threshold to remove depth pixel that are two far.

            is_fine: fine mode for the fragment fusion


            transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                every access. (default: :obj:`None`)

            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)
            num_random_pt: number of point we select when we test
        """
    def __init__(
            self,
            root,
            num_frame_per_fragment=50,
            mode="train_small",
            min_overlap_ratio=0.3,
            max_overlap_ratio=1.0,
            max_dist_overlap=0.01,
            tsdf_voxel_size=0.02,
            limit_size=700,
            depth_thresh=6,
            is_fine=True,
            transform=None,
            pre_transform=None,
            pre_transform_fragment=None,
            pre_filter=None,
            verbose=False,
            debug=False,
            is_online_matching=False,
            num_pos_pairs=1024,
            self_supervised=False,
            min_size_block=0.3,
            max_size_block=2,
            ss_transform=None,
            min_points=500,
    ):

        self.is_patch = False
        Base3DMatch.__init__(
            self,
            root,
            num_frame_per_fragment,
            mode,
            min_overlap_ratio,
            max_overlap_ratio,
            max_dist_overlap,
            tsdf_voxel_size,
            limit_size,
            depth_thresh,
            is_fine,
            transform,
            pre_transform,
            pre_transform_fragment,
            pre_filter,
            verbose,
            debug,
        )
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


class General3DMatchDataset(BaseSiameseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        pre_transform = self.pre_transform
        ss_transform = getattr(self, "ss_transform", None)
        train_transform = self.train_transform
        val_transform = self.val_transform
        test_transform = self.test_transform
        pre_filter = self.pre_filter
        test_pre_filter = getattr(self, "test_pre_filter", None)
        self.is_patch = dataset_opt.is_patch

        if dataset_opt.is_patch:
            self.train_dataset = Patch3DMatch(
                root=self._data_path,
                mode="train",
                radius_patch=dataset_opt.radius_patch,
                num_frame_per_fragment=dataset_opt.num_frame_per_fragment,
                max_dist_overlap=dataset_opt.max_dist_overlap,
                min_overlap_ratio=dataset_opt.min_overlap_ratio,
                tsdf_voxel_size=dataset_opt.tsdf_voxel_size,
                limit_size=dataset_opt.limit_size,
                depth_thresh=dataset_opt.depth_thresh,
                pre_transform=pre_transform,
                transform=train_transform,
                num_random_pt=dataset_opt.num_random_pt,
                is_offline=dataset_opt.is_offline,
                pre_filter=pre_filter,
            )

            self.val_dataset = Patch3DMatch(
                root=self._data_path,
                mode="val",
                radius_patch=dataset_opt.radius_patch,
                num_frame_per_fragment=dataset_opt.num_frame_per_fragment,
                max_dist_overlap=dataset_opt.max_dist_overlap,
                min_overlap_ratio=dataset_opt.min_overlap_ratio,
                tsdf_voxel_size=dataset_opt.tsdf_voxel_size,
                limit_size=dataset_opt.limit_size,
                depth_thresh=dataset_opt.depth_thresh,
                pre_transform=pre_transform,
                transform=val_transform,
                num_random_pt=dataset_opt.num_random_pt,
                is_offline=dataset_opt.is_offline,
                pre_filter=test_pre_filter,
            )
        else:

            self.train_dataset = Fragment3DMatch(
                root=self._data_path,
                mode="train",
                num_frame_per_fragment=dataset_opt.num_frame_per_fragment,
                max_dist_overlap=dataset_opt.max_dist_overlap,
                min_overlap_ratio=dataset_opt.min_overlap_ratio,
                tsdf_voxel_size=dataset_opt.tsdf_voxel_size,
                limit_size=dataset_opt.limit_size,
                depth_thresh=dataset_opt.depth_thresh,
                pre_transform=pre_transform,
                transform=train_transform,
                pre_filter=pre_filter,
                is_online_matching=dataset_opt.is_online_matching,
                num_pos_pairs=dataset_opt.num_pos_pairs,
                self_supervised=dataset_opt.self_supervised,
                min_size_block=dataset_opt.min_size_block,
                max_size_block=dataset_opt.max_size_block,
                ss_transform=ss_transform,
                min_points=dataset_opt.min_points)

            self.val_dataset = Fragment3DMatch(
                root=self._data_path,
                mode="val",
                num_frame_per_fragment=dataset_opt.num_frame_per_fragment,
                max_dist_overlap=dataset_opt.max_dist_overlap,
                min_overlap_ratio=dataset_opt.min_overlap_ratio,
                tsdf_voxel_size=dataset_opt.tsdf_voxel_size,
                limit_size=dataset_opt.limit_size,
                depth_thresh=dataset_opt.depth_thresh,
                pre_transform=pre_transform,
                transform=val_transform,
                is_online_matching=False,
                num_pos_pairs=dataset_opt.num_pos_pairs,
                self_supervised=False,
            )
            self.test_dataset = TestPair3DMatch(
                root=self._data_path,
                pre_transform=pre_transform,
                transform=test_transform,
                num_pos_pairs=50,
                max_dist_overlap=dataset_opt.max_dist_overlap
            )

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """
        Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        if self.is_patch:
            return PatchRegistrationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
        else:
            return FragmentRegistrationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
