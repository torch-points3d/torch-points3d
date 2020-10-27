"""
Code taken from
https://github.com/iralabdisco/point_clouds_registration_benchmark/blob/master/planetary_setup.py
"""
import gdown
import os
import os.path as osp
import logging
import requests
import glob
import re
import sys
import csv
import open3d
import numpy
import shutil

from zipfile import ZipFile
from ftplib import FTP

from torch_points3d.datasets.registration.basetest import BasePCRBTest
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.registration.base_siamese_dataset import BaseSiameseDataset

from torch_points3d.metrics.registration_tracker import FragmentRegistrationTracker

from torch_points3d.datasets.registration.utils import files_exist
from torch_points3d.datasets.registration.utils import makedirs

log = logging.getLogger(__name__)

class TestPairPlanetary(BasePCRBTest):
    DATASETS = [["p2at_met", "3dmap_dataset/p2at_met/p2at_met.zip"],
                ["box_met","3dmap_dataset/box_met/box_met.zip"]]

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
        self.link_pairs = "https://cloud.mines-paristech.fr/index.php/s/7cqiTMIIqwvMOtA/download"
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
                              use_fps=use_fps)

    def download(self):
        folder = osp.join(self.raw_dir, "test")
        if files_exist([folder]):  # pragma: no cover
            log.warning("already downloaded {}".format("test"))
            return
        else:
            makedirs(folder)
        ftp = FTP('asrl3.utias.utoronto.ca')
        ftp.login()
        log.info("Download elements in the file {}...".format(folder))
        for name, url in self.DATASETS:
            zip_file = osp.join(folder, name+'.zip')
            log.info("Downloading dataset %s" % name)
            ftp.retrbinary('RETR '+url, open(zip_file, 'wb').write)
            with ZipFile(zip_file, 'r') as zip_obj:
                log.info("Extracting dataset %s" % name)
                zip_obj.extractall(folder)
            with os.scandir(osp.join(folder, name)) as directory:
                log.info("Configuring dataset %s" % name)
                for entry in directory:
                    if entry.is_dir():
                        base_path = entry.path+"/"+entry.name
                        file_name = base_path+".xyz"
                        ground_truth_name = base_path+".gt"
                        pcd_file_name = entry.path+".pcd"
                        pcd = open3d.io.read_point_cloud(file_name, format="xyz",remove_nan_points=True, remove_infinite_points=True, print_progress=False)
                        ground_truth = numpy.loadtxt(ground_truth_name)
                        pcd.transform(ground_truth)
                        open3d.io.write_point_cloud(pcd_file_name, pcd, write_ascii=True, compressed=False, print_progress=False)
                        shutil.rmtree(entry.path)
            os.remove(zip_file)

        gdown.download("https://drive.google.com/uc?id=1marTTFGjlDTb-MLj7pm5zV1u-0IS-xFc", folder+"/p2at_met/box_map.pcd", quiet=True)
        self.download_pairs(folder)

    def process(self):
        super().process()



class PlanetaryDataset(BaseSiameseDataset):
    """
    this class is a dataset for testing registration algorithm on the Canadian Planetary dataset
    https://starslab.ca/enav-planetary-dataset/
    as defined in https://github.com/iralabdisco/point_clouds_registration_benchmark.
    """


    def __init__(self, dataset_opt):

        super().__init__(dataset_opt)
        pre_transform = self.pre_transform
        train_transform = self.train_transform
        ss_transform = getattr(self, "ss_transform", None)
        test_transform = self.test_transform

        # training is similar to test but only unsupervised training is allowed XD
        self.train_dataset = TestPairPlanetary(
            root=self._data_path,
            pre_transform=pre_transform,
            transform=train_transform,
            max_dist_overlap=dataset_opt.max_dist_overlap,
            self_supervised=True,
            min_size_block=dataset_opt.min_size_block,
            max_size_block=dataset_opt.max_size_block,
            num_pos_pairs=dataset_opt.num_pos_pairs,
            min_points=dataset_opt.min_points,
            ss_transform=ss_transform,
            use_fps=dataset_opt.use_fps
        )
        self.test_dataset = TestPairPlanetary(
            root=self._data_path,
            pre_transform=pre_transform,
            transform=test_transform,
            max_dist_overlap=dataset_opt.max_dist_overlap,
            num_pos_pairs=dataset_opt.num_pos_pairs,
            self_supervised=False)
