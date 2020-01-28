import numpy as np
import os
import os.path as osp
import torch
from src.datasets.registration.general3dmatch_dataset import General3DMatch
from src.core.data_transform import PatchExtractor
from torch_geometric.data import Batch

class Patch3DMatch(General3DMatch):

    def __init__(self, root,
                 radius_patch=0.3,
                 num_frame_per_fragment=50,
                 mode='train_small',
                 transform=None,
                 pre_transform=None,
                 pre_transform_fragment=None,
                 pre_filter=None,
                 verbose=False,
                 debug=False):
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
        `"BundleFusion: Real-time Globally Consistent 3D Reconstruction using Online Surface Re-integration
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
        """
        self.radius_patch = radius_patch
        super(Patch3DMatch, self).__init__(root,
                                           num_frame_per_fragment,
                                           mode,
                                           min_overlap_ratio=0.3,
                                           max_overlap_ratio=1.0,
                                           max_dist_overlap=0.1,
                                           transform,
                                           pre_transform,
                                           pre_filter,
                                           verbose,
                                           debug)

    def get(self, idx):
        match = np.load(osp.join(self.processed_dir, self.mode, 'filtered'))
        data_source = torch.load(match['path_source'])
        data_target = torch.load(match['path_target'])
        p_extractor = PatchExtractor(self.radius_patch)
        # select a random match on the list of match.
        # It cannot be 0 because matches are filtered.
        rand = np.random.randint(0, len(match['pair']))
        data_source = p_extractor(data_source, match['pair'][rand][0])
        data_target = p_extractor(data_target, match['pair'][rand][1])

        if(self.transform is not None):
            data_source = self.transform(data_source)
            data_target = self.transform(data_target)
