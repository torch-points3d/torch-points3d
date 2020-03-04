import json
import logging
import numpy as np
import os
import os.path as osp
from plyfile import PlyData
import shutil
import torch

from torch_geometric.data import Dataset, download_url, extract_zip
from torch_geometric.data import Data
from src.datasets.registration.detector import RandomDetector
from src.datasets.registration.utils import rgbd2fragment_rough
from src.datasets.registration.utils import rgbd2fragment_fine
from src.datasets.registration.utils import compute_overlap_and_matches
from src.datasets.registration.utils import to_list
from src.datasets.registration.utils import files_exist
from src.datasets.registration.utils import makedirs
from src.datasets.registration.utils import get_urls

log = logging.getLogger(__name__)


class Base3DMatch(Dataset):

    base = osp.abspath(osp.join(osp.realpath(__file__),
                                '..'))
    list_urls_train = get_urls(osp.join(base, 'urls', 'url_train.txt'))
    list_urls_train_small = get_urls(osp.join(base, 'urls', 'url_train_small.txt'))
    list_urls_train_tiny = get_urls(osp.join(base, 'urls', 'url_train_tiny.txt'))
    list_urls_val = get_urls(osp.join(base, 'urls', 'url_val.txt'))
    list_urls_test = get_urls(osp.join(base, 'urls', 'url_test.txt'))
    dict_urls = dict(train=list_urls_train,
                     train_small=list_urls_train_small,
                     train_tiny=list_urls_train_tiny,
                     val=list_urls_val,
                     test=list_urls_test)

    def __init__(self, root,
                 num_frame_per_fragment=50,
                 mode='train_small',
                 min_overlap_ratio=0.3,
                 max_overlap_ratio=1.0,
                 max_dist_overlap=0.01,
                 tsdf_voxel_size=0.01,
                 depth_thresh=6,
                 is_fine=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 verbose=False,
                 debug=False,
                 num_random_pt=5000):
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
        self.is_fine = is_fine
        self.num_frame_per_fragment = num_frame_per_fragment
        self.tsdf_voxel_size = tsdf_voxel_size
        self.depth_thresh = depth_thresh
        self.mode = mode
        self.num_random_pt = num_random_pt
        # select points for testing
        self.detector = RandomDetector(num_points=self.num_random_pt)
        # constant to compute overlap
        self.min_overlap_ratio = min_overlap_ratio
        self.max_overlap_ratio = max_overlap_ratio
        self.max_dist_overlap = max_dist_overlap
        if mode not in self.dict_urls.keys():
            raise RuntimeError('this mode {} does '
                               'not exist'
                               '(train_small|train_tiny|train|val|test)'.format(mode))
        super(Base3DMatch, self).__init__(root,
                                          transform,
                                          pre_transform,
                                          pre_filter)

        # path = self.processed_paths[0] if train else self.processed_paths[1]
        # self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [self.mode]

    @property
    def processed_file_names(self):
        return [osp.join(self.mode, 'fragment'),
                osp.join(self.mode, 'matches')]

    def download(self):

        if('test' not in self.mode):
            # we download the raw RGBD file for the train and the validation data
            folder = osp.join(self.raw_dir, self.mode)
            log.info("Download elements in the file {}...".format(folder))
            for url in self.dict_urls[self.mode]:
                path = download_url(url, folder, self.verbose)
                extract_zip(path, folder, self.verbose)
                os.unlink(path)
        else:
            # we download fragments for the test data available:
            # http://3dmatch.cs.princeton.edu/ section
            # Geometric Registration Benchmark
            folder_test = osp.join(self.raw_dir, self.mode)
            for url_raw in self.dict_urls[self.mode]:
                url = url_raw.split('\n')[0]
                path = download_url(url, folder_test)
                extract_zip(path, folder_test)
                print(path)
                folder = path.split('.zip')[0]
                os.unlink(path)
                path_eval = download_url(url.split('.zip')[0]+'-evaluation.zip', folder)
                extract_zip(path_eval, folder)
                os.unlink(path_eval)
                folder_eval = path_eval.split('.zip')[0]
                for f in os.listdir(folder_eval):
                    os.rename(osp.join(folder_eval, f), osp.join(folder, f))
                shutil.rmtree(folder_eval)


    def _create_fragment(self, mod):
        r"""
        create fragments from the rgbd frames ie a partial reconstruction of
        the scene with some frames(usually 50).
        We will only use the first sequence for each scene
        """

        print("Create fragment from RGBD frames...")
        if files_exist([osp.join(self.processed_dir, mod, 'fragment')]):  # pragma: no cover
            log.warning("the fragments on mode {} already exists".format(mod))
            return
        for scene_path in os.listdir(osp.join(self.raw_dir, mod)):
            # TODO list the right sequences.
            list_seq = [f for f in os.listdir(osp.join(self.raw_dir, mod,
                                                       scene_path)) if 'seq' in f]
            for seq in list_seq:
                frames_dir = osp.join(self.raw_dir, self.mode,
                                      scene_path, seq)
                out_dir = osp.join(self.processed_dir,
                                   mod, 'fragment',
                                   scene_path, seq)
                makedirs(out_dir)
                path_intrinsic = osp.join(self.raw_dir,
                                          self.mode, scene_path,
                                          'camera-intrinsics.txt')
                list_path_frames = sorted([osp.join(frames_dir, f)
                                           for f in os.listdir(frames_dir)
                                           if 'png' in f and 'depth' in f])
                # list_path_color = sorted([osp.join(frames_dir, f)
                #                          for f in os.listdir(frames_dir)
                #                          if 'png' in f and 'color' in f])
                list_path_trans = sorted([osp.join(frames_dir, f)
                                          for f in os.listdir(frames_dir)
                                          if 'pose' in f and 'txt' in f])
                # compute each fragment and save it
                if(not self.is_fine):
                    rgbd2fragment_rough(list_path_frames, path_intrinsic,
                                        list_path_trans, out_dir,
                                        self.num_frame_per_fragment,
                                        pre_transform=self.pre_transform)
                else:
                    assert len(list_path_frames) == len(list_path_trans), \
                        log.error("For the sequence {},"
                                  "the number of frame "
                                  "and the number of "
                                  "pose is different".format(frames_dir))
                    rgbd2fragment_fine(list_path_frames,
                                       path_intrinsic,
                                       list_path_trans,
                                       out_dir, self.num_frame_per_fragment,
                                       voxel_size=self.tsdf_voxel_size,
                                       pre_transform=self.pre_transform,
                                       depth_thresh=self.depth_thresh)

    def _compute_matches_between_fragments(self, mod):

        out_dir = osp.join(self.processed_dir,
                           mod, 'matches')
        if files_exist([out_dir]):  # pragma: no cover
            return
        makedirs(out_dir)

        for scene_path in os.listdir(osp.join(self.raw_dir, mod)):

            list_seq = sorted([f for f in os.listdir(osp.join(self.raw_dir, mod,
                                                              scene_path)) if 'seq' in f])
            for seq in list_seq:
                log.info("{}, {}".format(scene_path, seq))
                fragment_dir = osp.join(self.processed_dir,
                                        mod, 'fragment',
                                        scene_path, seq)
                list_fragment_path = sorted([osp.join(fragment_dir, f)
                                             for f in os.listdir(fragment_dir)
                                             if 'fragment' in f])
                log.info("compute_overlap_and_matches")
                ind = 0
                for path1 in list_fragment_path:
                    for path2 in list_fragment_path:
                        if path1 < path2:
                            out_path = osp.join(out_dir,
                                                'matches{:06d}.npy'.format(ind))

                            match = compute_overlap_and_matches(
                                path1, path2, self.max_dist_overlap)
                            if(self.verbose):
                                log.info(match['path_source'],
                                         match['path_target'],
                                         'overlap={}'.format(match['overlap']))
                            if(np.max(match['overlap']) > self.min_overlap_ratio and
                               np.max(match['overlap']) < self.max_overlap_ratio):
                                np.save(out_path, match)
                                ind += 1

    def _compute_points_on_fragments(self, mod):
        """
        compute descriptors on fragments and save fragments with points on a pt file.
        """
        out_dir = osp.join(self.processed_dir,
                           mod, 'fragment')
        if files_exist([out_dir]):  # pragma: no cover
            return
        makedirs(out_dir)
        ind = 0
        # table to map fragment numper with
        self.table = dict()
        tot = 0
        for scene_path in os.listdir(osp.join(self.raw_dir, mod)):

            fragment_dir = osp.join(self.raw_dir,
                                    mod,
                                    scene_path)
            list_fragment_path = sorted([osp.join(fragment_dir, f)
                                         for f in os.listdir(fragment_dir)
                                         if 'ply' in f])

            for i, fragment_path in enumerate(list_fragment_path):
                out_path = osp.join(out_dir, 'fragment_{:06d}.pt'.format(ind))

                # read ply file
                with open(fragment_path, 'rb') as f:
                    data = PlyData.read(f)
                pos = ([torch.tensor(data['vertex'][axis]) for axis in ['x', 'y', 'z']])
                pos = torch.stack(pos, dim=-1)

                # compute keypoints indices
                data = Data(pos=pos)
                if(self.pre_transform is not None):
                    data = self.pre_transform(data)
                data = self.detector(data)
                torch.save(data, out_path)
                num_points = getattr(data, 'keypoints', data.pos).shape[0]
                tot += num_points
                self.table[ind] = {'in_path': fragment_path,
                                   'scene_path': scene_path,
                                   'out_path': out_path,
                                   'num_points': num_points}
                ind += 1
        self.table['total_num_points'] = tot

        # save this file into json
        with open(osp.join(out_dir, 'table.txt'), 'w') as f:
            json.dump(self.table, f)

    def process(self):
        if('test' not in self.mode):
            log.info("create fragments")
            self._create_fragment(self.mode)
            log.info("compute matches")
            self._compute_matches_between_fragments(self.mode)
        else:
            self._compute_points_on_fragments(self.mode)

    def get(self, idx):
        raise NotImplementedError("implement class to get patch or fragment or more")
