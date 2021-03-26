"""
Code taken from
https://github.com/iralabdisco/point_clouds_registration_benchmark/blob/master/tum_setup.py
"""
import gdown
import os
import logging
import sys

from zipfile import ZipFile

from torch_points3d.datasets.registration.basetest import BasePCRBTest
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.registration.base_siamese_dataset import BaseSiameseDataset

# from torch_points3d.metrics.registration_tracker import FragmentRegistrationTracker

from torch_points3d.datasets.registration.utils import files_exist
from torch_points3d.datasets.registration.utils import makedirs

log = logging.getLogger(__name__)

class TestPairTUM(BasePCRBTest):
    DATASETS = [["long_office_household","https://drive.google.com/uc?id=1Uy9aUyZbjW26-lZ1oyqOUxJ9tP7E-fWP"],
        ["pioneer_slam","https://drive.google.com/uc?id=1ha3rxXewWrlCv6SPbpg21I8Wpl9v9fi1"],
        ["pioneer_slam3","https://drive.google.com/uc?id=1L8FzuFf1Nc3hy6YfhHYM3qSUKmP_eMBG"]]

    def __init__(self, root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 verbose=False,
                 debug=False,
                 num_pos_pairs=200,
                 max_dist_overlap=0.01,
                 self_supervised=False,
                 min_size_block=2,
                 max_size_block=3,
                 min_points=500,
                 ss_transform=None,
                 use_fps=False):
        self.link_pairs = "https://cloud.mines-paristech.fr/index.php/s/yjd20Ih9ExqLlHM/download"
        BasePCRBTest.__init__(self,
                              root=root,
                              transform=transform,
                              pre_transform=pre_transform,
                              pre_filter=pre_filter,
                              verbose=verbose, debug=debug,
                              max_dist_overlap=max_dist_overlap,
                              num_pos_pairs=num_pos_pairs,
                              self_supervised=self_supervised,
                              min_size_block=min_size_block,
                              max_size_block=max_size_block,
                              min_points=min_points,
                              ss_transform=ss_transform,
                              use_fps=use_fps,
                              is_name_path_int=False)

    def download(self):
        folder = os.path.join(self.raw_dir, "test")
        if files_exist([folder]):  # pragma: no cover
            log.warning("already downloaded {}".format("test"))
            return
        else:
            makedirs(folder)
        log.info("Download elements in the file {}...".format(folder))
        for name, url in self.DATASETS:
            dataset_folder = os.path.join(folder,name)
            os.mkdir(dataset_folder)
            log.info(f'Downloading sequence {name}')
            filename = os.path.join(folder,name+".zip")
            gdown.download(url, filename, quiet=False)
            with ZipFile(filename, 'r') as zip_obj:
                zip_obj.extractall(dataset_folder)
            os.remove(filename)

        self.download_pairs(folder)

    def process(self):
        super().process()


class TUMDataset(BaseSiameseDataset):
    """
    this class is a dataset for testing registration algorithm on the TUM RGB-D Dataset
    https://vision.in.tum.de/data/datasets/rgbd-dataset
    as defined in https://github.com/iralabdisco/point_clouds_registration_benchmark.
    """


    def __init__(self, dataset_opt):

        super().__init__(dataset_opt)
        pre_transform = self.pre_transform
        train_transform = self.train_transform
        ss_transform = getattr(self, "ss_transform", None)
        test_transform = self.test_transform

        # training is similar to test but only unsupervised training is allowed XD
        self.train_dataset = TestPairTUM(root=self._data_path,
                                         pre_transform=pre_transform,
                                         transform=train_transform,
                                         max_dist_overlap=dataset_opt.max_dist_overlap,
                                         self_supervised=True,
                                         min_size_block=dataset_opt.min_size_block,
                                         max_size_block=dataset_opt.max_size_block,
                                         num_pos_pairs=dataset_opt.num_pos_pairs,
                                         min_points=dataset_opt.min_points,
                                         ss_transform=ss_transform,
                                         use_fps=dataset_opt.use_fps)
        self.test_dataset = TestPairTUM(root=self._data_path,
                                        pre_transform=pre_transform,
                                        transform=test_transform,
                                        max_dist_overlap=dataset_opt.max_dist_overlap,
                                        num_pos_pairs=dataset_opt.num_pos_pairs,
                                        self_supervised=False)
