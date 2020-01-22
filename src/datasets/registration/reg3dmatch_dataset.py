import collections
import errno
import logging
import os
import os.path as osp
import shutil
import torch
from torch_geometric.data import Dataset, Data, download_url, extract_zip
from torch_geometric.data import DataLoader

from src.datasets.base_dataset import BaseDataset
from src.datasets.registration.utils import rgbd2fragment

log = logging.getLogger(__name__)


def to_list(x):
    """
    taken from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/dataset.html#Dataset
    """
    if not isinstance(x, collections.Iterable) or isinstance(x, str):
        x = [x]
    return x


def files_exist(files):
    """
    taken from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/dataset.html#Dataset
    """

    return all([osp.exists(f) for f in files])


def makedirs(path):
    """
    taken from https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/makedirs.py
    """
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def get_urls(filename):
    res = []
    with open(filename, 'r') as f:
        res = f.readlines()
    return res


class General3DMatch(Dataset):

    list_urls_train = get_urls(osp.join('urls', 'url_train.txt'))
    list_urls_train_small = get_urls(osp.join('urls', 'url_train_small.txt'))
    list_urls_test = get_urls(osp.join('urls', 'url_test.txt'))
    dict_urls = dict(train=list_urls_train,
                     train_small=list_urls_train_small,
                     test=list_urls_test)

    def __init__(self, root,
                 num_frame_per_fragment=50,
                 mode='train_small',
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

        self.verbose = verbose
        self.debug = debug

        self.num_frame_per_fragment = num_frame_per_fragment
        self.mode = mode
        if mode not in self.dict_urls.keys():
            raise RuntimeError('this mode {} does '
                               'not exist'
                               '(train_small|train|test)'.format(mode))
        super(General3DMatch, self).__init__(root,
                                             transform,
                                             pre_transform,
                                             pre_filter)

        # path = self.processed_paths[0] if train else self.processed_paths[1]
        # self.data, self.slices = torch.load(path)

    @property
    def fragment_dir(self):
        return osp.join(self.root, 'fragment')

    @property
    def fragment_paths(self):
        r"""The filepaths to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        files = to_list(self.fragment_file_names)
        return [osp.join(self.fragment_dir, f) for f in files]

    @property
    def raw_file_names(self):
        return self.mode

    @property
    def processed_file_names(self):
        return self.mode

    @property
    def fragment_file_names(self):
        return self.mode

    def download(self):
        folder = osp.join(self.raw_dir, self.mode)
        log.info("Download elements in the file {}...".format(folder))
        for url in self.dict_urls[self.mode]:
            path = download_url(url, folder, self.verbose)
            extract_zip(path, folder, self.verbose)
            os.unlink(path)

    def _create_fragment(self):
        r"""
        create fragments from the rgbd frames ie a partial reconstruction of
        the scene with some frames(usually 50).
        We will only use the first sequence for each scene
        """
        if files_exist(self.fragment_paths):  # pragma: no cover
            return
        print("Create fragment from RGBD frames...")
        makedirs(self.fragment_paths)
        for scene_path in os.listdir(osp.join(self.raw_dir, self.mode)):
            frames_dir = osp.join(self.raw_dir, self.mode,
                                  scene_path, 'seq_01')
            out_dir = osp.join(self.fragment_dir, self.mode, scene_path)
            path_intrinsic = osp.join(self.raw_dir,
                                      self.mode, scene_path,
                                      'camera-intrinsics.txt')
            list_path_frames = sorted([osp.join(frames_dir, f)
                                       for f in osp.listdir(frames_dir)
                                       if 'png' in f and 'depth' in f])
            list_path_trans = sorted([osp.join(frames_dir, f)
                                      for f in osp.listdir(frames_dir)
                                      if 'pose' in f and 'txt' in f])
            # compute each fragment and save it
            rgbd2fragment(list_path_frames, path_intrinsic,
                          list_path_trans, out_dir,
                          self.num_frame_per_fragment)




    def compute_and_save(self):
        raise NotImplementedError("Implement here the computation of the "
                                  "training set"
                                  "")

    def process(self):
        self._create_fragment()
        self.compute_and_save()

    def get(self, idx):
        pass
