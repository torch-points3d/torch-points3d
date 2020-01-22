import logging
import os
import os.path as osp
import shutil
import torch
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from torch_geometric.data import DataLoader

from .base_dataset import BaseDataset

log = logging.getLogger(__name__)

def create_fragment():
    pass

def get_urls(filename):
    res = []
    with open(filename, 'r') as f:
        res = f.readlines()
    return res


class Descr3DMatch(InMemoryDataset):

    list_urls_sun3d = get_urls(
        osp.join('urls', 'url_sun3d.txt'))
    list_urls_7_scenes = get_urls(
        osp.join('urls', 'url_7-scenes.txt'))
    list_urls_analysis_by_synthesis = get_urls(
        osp.join('urls', 'url_analysis-by-synthesis.txt'))
    list_urls_bundlefusion = get_urls(
        osp.join('urls', 'url_bundlefusion.txt'))
    list_urls_rgbd_scene_v2 = get_urls(
        osp.join('urls', 'url_rgbd-scenes-v2.txt'))
    dict_urls = dict(sun3d=list_urls_sun3d,
                     seven_scenes=list_urls_7_scenes,
                     analysis_by_synthesis=list_urls_analysis_by_synthesis,
                     bundlefusion=list_urls_bundlefusion,
                     rgbd_scene_v2=list_urls_rgbd_scene_v2)

    def __init__(self, root,
                 scenes=['sun3d'],
                 num_frame_per_fragment=50,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 verbose=False,
                 debug=False):
        r"""
        the Princeton 3DMatch dataset from the
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

            scenes (list, optional): list of string which indicate which
                dataset we should download. (default: obj:`['sun3d']`)

            num_frame_per_fragment (int, optional): indicate the number of frames
                we use to build fragments. If it is equal to 0, then we don't
                build fragments and use the raw frames.

            train (bool, optional): If :obj:`True`, loads the training dataset,
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

        self.verbose = verbose
        self.debug = debug

        self.num_frame_per_fragment = num_frame_per_fragment
        self.scenes = scenes
        super(Descr3DMatch, self).__init__(root,
                                           transform,
                                           pre_transform,
                                           pre_filter)

        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return self.scenes

    @property
    def processed_file_names(self):
        pass

    def download(self):
        for scene in self.scenes:
            if(self.verbose):
                log.info("Download elements of the scene {}...".format(scene))
            folder = osp.join(self.raw_dir, scene)
            try:
                for url in self.dict_urls[scene]:
                    path = download_url(url, folder, self.verbose)
                    extract_zip(path, folder, self.verbose)
                    os.unlink(path)
            except KeyError:
                log.error("Could not download this file: {} because"
                          "{} doesn't belong to"
                          "3DMatch Dataset".format(scene, scene))

    def create_fragment(self):
        r"""
        create fragments from the rgbd frames ie a partial reconstruction of
        the scene with some frames(usually 50).
        """

    def select_patch_from_fragment(self):
        pass

    def select_patch_from_frame(self):
        pass

    def process(self):
        pass
