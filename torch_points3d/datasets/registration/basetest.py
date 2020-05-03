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

from torch_points3d.datasets.registration.utils import rgbd2fragment_rough
from torch_points3d.datasets.registration.utils import rgbd2fragment_fine
from torch_points3d.datasets.registration.utils import compute_overlap_and_matches
from torch_points3d.datasets.registration.utils import to_list
from torch_points3d.datasets.registration.utils import files_exist
from torch_points3d.datasets.registration.utils import makedirs
from torch_points3d.datasets.registration.utils import get_urls
from torch_points3d.datasets.registration.utils import PatchExtractor


log = logging.getLogger(__name__)


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
                 num_random_pt=5000):
        """
        a baseDataset that download a dataset,
        apply preprocessing, and compute keypoints
        """

        self.num_random_pt = num_random_pt
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
        out_dir = osp.join(self.processed_dir,
                           'fragment')
        if files_exist([out_dir]):  # pragma: no cover
            return
        makedirs(out_dir)
        ind = 0
        # table to map fragment numper with
        self.table = dict()

        for scene_path in os.listdir(osp.join(self.raw_dir, "raw_fragment")):

            fragment_dir = osp.join(self.raw_dir,
                                    "raw_fragment",
                                    scene_path)
            list_fragment_path = sorted([f
                                         for f in os.listdir(fragment_dir)
                                         if 'ply' in f])

            for i, f_p in enumerate(list_fragment_path):
                fragment_path = osp.join(fragment_dir, f_p)
                out_path = osp.join(out_dir, 'fragment_{:06d}.pt'.format(ind))

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

    def process(self):
        self._pre_transform_fragments_ply()

    def __getitem__(self, idx):
        raise NotImplementedError("implement class to get patch or fragment or more")


class Base3DMatchTest(BaseTest):

    def __init__(self, root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 verbose=False,
                 debug=False,
                 num_random_pt=5000):
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
                                              num_random_pt)

    def download(self):
        folder_test = osp.join(self.raw_dir, 'raw_fragment')
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
                 num_random_pt=5000):
        """
        Base for ETH Dataset. The main goal is to see
        if the descriptors generalize well.
        """

        self.list_urls_test = ["url"]
        super(BaseTest, self).__init__(root,
                                       transform,
                                       pre_transform,
                                       pre_filter,
                                       verbose,
                                       debug,
                                       num_random_pt)

    def download(self):
        raise NotImplementedError("need to implement test for this dataset")
