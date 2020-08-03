import json
import logging
import numpy as np
import os
import os.path as osp
from plyfile import PlyData
import shutil
import torch
import re

from torch_geometric.data import Dataset, download_url, extract_zip
from torch_geometric.data import Data

from torch_points3d.datasets.registration.utils import rgbd2fragment_rough
from torch_points3d.datasets.registration.utils import rgbd2fragment_fine
from torch_points3d.datasets.registration.utils import compute_overlap_and_matches
from torch_points3d.datasets.registration.utils import to_list
from torch_points3d.datasets.registration.utils import files_exist
from torch_points3d.datasets.registration.utils import makedirs
from torch_points3d.datasets.registration.utils import get_urls
from torch_points3d.datasets.registration.utils import PatchExtractor


log = logging.getLogger(__name__)


def find_int(path):
    number = re.findall("\d+", path)[0]
    return int(number)


def read_gt_log(path):
    """
    read the gt.log of evaluation set of 3DMatch or ETH Dataset and parse it.
    """
    list_pair = []
    list_mat = []
    with open(path, "r") as f:
        all_mat = f.readlines()
    mat = np.zeros((4, 4))
    for i in range(len(all_mat)):
        if i % 5 == 0:
            if i != 0:
                list_mat.append(mat)
            mat = np.zeros((4, 4))
            list_pair.append(list(map(int, all_mat[i].split("\t")[:-1])))
        else:
            line = all_mat[i].split("\t")

            mat[i % 5 - 1] = np.asarray(line[:4], dtype=np.float)
    list_mat.append(mat)
    return list_pair, list_mat


class SimplePatch(torch.utils.data.Dataset):

    def __init__(self, list_patches, transform=None):
        """
        transform a list of Data into a dataset(and apply transform)
        """
        if(transform is None):
            self.list_patches = list_patches
        else:
            self.list_patches = [transform(p) for p in list_patches]
        self.transform = transform

    def __len__(self):
        return len(self.list_patches)

    def __getitem__(self, idx):
        data = self.list_patches[idx]
        return data

    @property
    def num_features(self):
        if self[0].x is None:
            return 0
        return 1 if self[0].x.dim() == 1 else self[0].x.size(1)


class BaseTest(Dataset):

    def __init__(self, root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 verbose=False,
                 debug=False,
                 max_dist_overlap=0.01):
        """
        a baseDataset that download a dataset,
        apply preprocessing, and compute keypoints
        """
        self.max_dist_overlap = max_dist_overlap
        super(BaseTest, self).__init__(root,
                                       transform,
                                       pre_transform,
                                       pre_filter)

    @property
    def raw_file_names(self):
        return ["raw_fragment"]

    @property
    def processed_file_names(self):
        return ["fragment"]

    def download(self):
        raise NotImplementedError('need to download the dataset')

    def _pre_transform_fragments_ply(self):
        """
        apply pre_transform on fragments (ply) and save the results
        """
        out_dir = osp.join(self.processed_dir, 'test',
                           'fragment')
        if files_exist([out_dir]):  # pragma: no cover
            return
        makedirs(out_dir)
        ind = 0
        # table to map fragment numper with
        self.table = dict()

        for scene_path in os.listdir(osp.join(self.raw_dir, "test")):

            fragment_dir = osp.join(self.raw_dir,
                                    "test",
                                    scene_path)
            list_fragment_path = sorted([f
                                         for f in os.listdir(fragment_dir)
                                         if 'ply' in f])

            for i, f_p in enumerate(list_fragment_path):
                fragment_path = osp.join(fragment_dir, f_p)
                out_dir = osp.join(self.processed_dir, "test",
                                   'fragment', scene_path)
                makedirs(out_dir)
                out_path = osp.join(out_dir,
                                    'fragment_{:06d}.pt'.format(find_int(f_p)))
                # read ply file
                with open(fragment_path, 'rb') as f:
                    data = PlyData.read(f)
                pos = ([torch.tensor(data['vertex'][axis]) for axis in ['x', 'y', 'z']])
                pos = torch.stack(pos, dim=-1)
                data = Data(pos=pos)
                if(self.pre_transform is not None):
                    data = self.pre_transform(data)
                torch.save(data, out_path)
                self.table[ind] = {'in_path': fragment_path,
                                   'scene_path': scene_path,
                                   'fragment_name': f_p,
                                   'out_path': out_path}
                ind += 1

        # save this file into json
        with open(osp.join(out_dir, 'table.json'), 'w') as f:
            json.dump(self.table, f)

    def _compute_matches_between_fragments(self):
        ind = 0
        out_dir = osp.join(self.processed_dir,
                           "test", "matches")
        if files_exist([out_dir]):  # pragma: no cover
            return
        makedirs(out_dir)

        list_scene = os.listdir(osp.join(self.raw_dir, "test"))
        for scene in list_scene:
            path_log = osp.join(self.raw_dir, "test", scene, "gt.log")
            list_pair_num, list_mat = read_gt_log(path_log)
            for i, pair in enumerate(list_pair_num):
                path1 = osp.join(self.processed_dir, "test",
                                 'fragment', scene,
                                 'fragment_{:06d}.pt'.format(pair[0]))
                path2 = osp.join(self.processed_dir, "test",
                                 'fragment', scene,
                                 'fragment_{:06d}.pt'.format(pair[1]))
                data1 = torch.load(path1)
                data2 = torch.load(path2)
                match = compute_overlap_and_matches(
                    data1, data2, self.max_dist_overlap,
                    trans_gt=torch.from_numpy(np.linalg.inv(list_mat[i])).to(data1.pos.dtype))
                match['path_source'] = path1
                match['path_target'] = path2
                match['name_source'] = str(pair[0])
                match['name_target'] = str(pair[1])
                match['scene'] = scene
                out_path = osp.join(
                    self.processed_dir, "test",
                    'matches',
                    'matches{:06d}.npy'.format(ind))
                np.save(out_path, match)
                ind += 1

    def process(self):
        self._pre_transform_fragments_ply()
        self._compute_matches_between_fragments()

    def __getitem__(self, idx):
        raise NotImplementedError("implement class to get patch or fragment or more")


class Base3DMatchTest(BaseTest):

    def __init__(self, root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 verbose=False,
                 debug=False,
                 max_dist_overlap=0.01):
        """
        Base 3D Match but for testing
        """
        base = osp.abspath(osp.join(osp.realpath(__file__),
                                    '..'))
        self.list_urls_test = get_urls(osp.join(base, 'urls', 'url_test.txt'))
        super(Base3DMatchTest, self).__init__(root,
                                              transform,
                                              pre_transform,
                                              pre_filter,
                                              verbose,
                                              debug,
                                              max_dist_overlap)

    def download(self):
        folder_test = osp.join(self.raw_dir, 'test')
        if files_exist([folder_test]):  # pragma: no cover
            log.warning("already downloaded {}".format('test'))
            return
        for url_raw in self.list_urls_test:
            url = url_raw.split('\n')[0]
            path = download_url(url, folder_test)
            extract_zip(path, folder_test)
            log.info(path)
            folder = path.split('.zip')[0]
            os.unlink(path)
            path_eval = download_url(url.split('.zip')[0]+'-evaluation.zip',
                                     folder)
            extract_zip(path_eval, folder)
            os.unlink(path_eval)
            folder_eval = path_eval.split('.zip')[0]
            for f in os.listdir(folder_eval):
                os.rename(osp.join(folder_eval, f), osp.join(folder, f))
            shutil.rmtree(folder_eval)


class BaseETHTest(BaseTest):

    def __init__(self, root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 verbose=False,
                 debug=False,
                 num_random_pt=5000,
                 max_dist_overlap=0.01):
        """
        Base for ETH Dataset. The main goal is to see
        if the descriptors generalize well.
        """
        self.num_random_pt = num_random_pt
        super(BaseTest, self).__init__(root,
                                       transform,
                                       pre_transform,
                                       pre_filter,
                                       verbose,
                                       debug,
                                       max_dist_overlap)

    def download(self):
        pass
