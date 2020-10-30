"""
Code taken from
https://github.com/iralabdisco/point_clouds_registration_benchmark/blob/master/eth_setup.py
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

from zipfile import ZipFile


from torch_points3d.datasets.registration.basetest import BasePCRBTest
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.registration.base_siamese_dataset import BaseSiameseDataset

from torch_points3d.metrics.registration_tracker import FragmentRegistrationTracker

from torch_points3d.datasets.registration.utils import files_exist
from torch_points3d.datasets.registration.utils import makedirs

log = logging.getLogger(__name__)


def asl_to_pcd(folder_name):
    pattern = re.compile("PointCloud(\d*).csv")

    for filename in os.listdir(folder_name):
        matched_string = pattern.match(filename)
        full_filename = folder_name+"/"+filename
        if matched_string:
            points = []
            with open(full_filename) as csv_cloud:
                csv_reader = csv.reader(csv_cloud, delimiter=',')
                line = 0
                out_filename = folder_name+"/"+"PointCloud"+matched_string.group(1)+".pcd"
                for row in csv_reader:
                    if line != 0:
                        points.append([float(row[1]),float(row[2]),float(row[3])])
                    else:
                        line=line+1
            with open(out_filename, "w") as out_file:
                out_file.write("# .PCD v.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH "+str(len(points))+"\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS "+str(len(points))+"\nDATA ascii")
                for point in points:
                    out_file.write("\n"+str(point[0])+" "+str(point[1])+" "+str(point[2]))


class TestPairETH(BasePCRBTest):
    DATASETS = [["apartment", "http://robotics.ethz.ch/~asl-datasets/apartment_03-Dec-2011-18_13_33/csv_global/global_frame.zip"],
                ["hauptgebaude", "http://robotics.ethz.ch/~asl-datasets/ETH_hauptgebaude_23-Aug-2011-18_43_49/csv_global/global_frame.zip"],
                ["stairs", "http://robotics.ethz.ch/~asl-datasets/stairs_26-Aug-2011-14_26_14/csv_global/global_frame.zip"],
                ["plain", "http://robotics.ethz.ch/~asl-datasets/plain_01-Sep-2011-16_39_18/csv_global/global_frame.zip"],
                ["gazebo_summer", "http://robotics.ethz.ch/~asl-datasets/gazebo_summer_04-Aug-2011-16_13_22/csv_global/global_frame.zip"],
                ["gazebo_winter", "http://robotics.ethz.ch/~asl-datasets/gazebo_winter_18-Jan-2012-16_10_04/csv_global/global_frame.zip"],
                ["wood_summer", "http://robotics.ethz.ch/~asl-datasets/wood_summer_25-Aug-2011-13_00_30/csv_global/global_frame.zip"],
                ["wood_autumn", "http://robotics.ethz.ch/~asl-datasets/wood_autumn_09-Dec-2011-15_44_05/csv_global/global_frame.zip"]]

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
        self.link_pairs = "https://cloud.mines-paristech.fr/index.php/s/aIRBieRybts3kEs/download"
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
        print(folder)
        if files_exist([folder]):  # pragma: no cover
            log.warning("already downloaded {}".format("test"))
            return
        else:
            makedirs(folder)
        log.info("Download elements in the file {}...".format(folder))
        for name, url in self.DATASETS:
            req = requests.get(url)
            with open(osp.join(folder, name+".zip"), "wb") as archive:
                archive.write(req.content)
            with ZipFile(osp.join(folder, name+".zip"), "r") as zip_obj:
                log.info("extracting dataset {}".format(name))
                zip_obj.extractall(osp.join(folder, name))
                log.info("converting to PCD")
                asl_to_pcd(osp.join(folder, name))
            file_not_to_remove = glob.glob(osp.join(folder, name, "*.pcd"))
            filelist = glob.glob(osp.join(folder, name, "*"))
            for file_to_remove in filelist:
                if file_to_remove not in file_not_to_remove:
                    os.remove(file_to_remove)
            os.remove(osp.join(folder, name+".zip"))
        self.download_pairs(folder)

    def process(self):
        super().process()



class ETHDataset(BaseSiameseDataset):
    """
    this class is a dataset for testing registration algorithm on ETH dataset
    https://projects.asl.ethz.ch/datasets/doku.php?id=laserregistration:laserregistration
    as defined in https://github.com/iralabdisco/point_clouds_registration_benchmark.
    """


    def __init__(self, dataset_opt):

        super().__init__(dataset_opt)
        pre_transform = self.pre_transform
        train_transform = self.train_transform
        ss_transform = getattr(self, "ss_transform", None)
        test_transform = self.test_transform

        # training is similar to test but only unsupervised training is allowed XD
        self.train_dataset = TestPairETH(root=self._data_path,
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
        self.test_dataset = TestPairETH(root=self._data_path,
                                        pre_transform=pre_transform,
                                        transform=test_transform,
                                        max_dist_overlap=dataset_opt.max_dist_overlap,
                                        num_pos_pairs=dataset_opt.num_pos_pairs,
                                        self_supervised=False)
