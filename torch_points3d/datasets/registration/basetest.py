import json
import logging
import numpy as np
import pandas as pd
import os
import os.path as osp
from plyfile import PlyData
import shutil
import torch
import re
import requests
from zipfile import ZipFile
import random

from torch_geometric.data import Dataset, download_url, extract_zip
from torch_geometric.data import Data

from torch_points3d.core.data_transform import FixedSphereDropout
from torch_points3d.datasets.registration.base_siamese_dataset import GeneralFragment

from torch_points3d.datasets.registration.utils import rgbd2fragment_rough
from torch_points3d.datasets.registration.utils import rgbd2fragment_fine
from torch_points3d.datasets.registration.utils import compute_overlap_and_matches
from torch_points3d.datasets.registration.utils import to_list
from torch_points3d.datasets.registration.utils import files_exist
from torch_points3d.datasets.registration.utils import makedirs
from torch_points3d.datasets.registration.utils import get_urls
from torch_points3d.datasets.registration.utils import PatchExtractor

from torch_points3d.datasets.registration.pair import Pair, MultiScalePair
from torch_points3d.datasets.registration.utils import tracked_matches

from torch_points_kernels.points_cpu import dense_knn

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
        if transform is None:
            self.list_patches = list_patches
        else:
            self.list_patches = [transform(p) for p in list_patches]
        self.transform = transform

    def __len__(self):
        return len(self.list_patches)

    def len(self):
        return len(self)

    def __getitem__(self, idx):
        data = self.list_patches[idx]
        return data

    @property
    def num_features(self):
        if self[0].x is None:
            return 0
        return 1 if self[0].x.dim() == 1 else self[0].x.size(1)


class Base3DMatchTest(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        verbose=False,
        debug=False,
        max_dist_overlap=0.01,
    ):
        """
        a baseDataset that download a dataset,
        apply preprocessing, and compute keypoints
        """
        self.max_dist_overlap = max_dist_overlap
        base = osp.abspath(osp.join(osp.realpath(__file__), ".."))
        self.list_urls_test = get_urls(osp.join(base, "urls", "url_test.txt"))
        super(Base3DMatchTest, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ["raw_fragment"]

    @property
    def processed_file_names(self):
        return ["fragment"]

    def download(self):
        folder_test = osp.join(self.raw_dir, "test")
        if files_exist([folder_test]):  # pragma: no cover
            log.warning("already downloaded {}".format("test"))
            return
        for url_raw in self.list_urls_test:
            url = url_raw.split("\n")[0]
            path = download_url(url, folder_test)
            extract_zip(path, folder_test)
            log.info(path)
            folder = path.split(".zip")[0]
            os.unlink(path)
            path_eval = download_url(url.split(".zip")[0] + "-evaluation.zip", folder)
            extract_zip(path_eval, folder)
            os.unlink(path_eval)
            folder_eval = path_eval.split(".zip")[0]
            for f in os.listdir(folder_eval):
                os.rename(osp.join(folder_eval, f), osp.join(folder, f))
            shutil.rmtree(folder_eval)

    def _pre_transform_fragments_ply(self):
        """
        apply pre_transform on fragments (ply) and save the results
        """
        out_dir = osp.join(self.processed_dir, "test", "fragment")
        if files_exist([out_dir]):  # pragma: no cover
            return
        makedirs(out_dir)
        ind = 0
        # table to map fragment numper with
        self.table = dict()

        for scene_path in os.listdir(osp.join(self.raw_dir, "test")):

            fragment_dir = osp.join(self.raw_dir, "test", scene_path)
            list_fragment_path = sorted([f for f in os.listdir(fragment_dir) if "ply" in f])

            for i, f_p in enumerate(list_fragment_path):
                fragment_path = osp.join(fragment_dir, f_p)
                out_dir = osp.join(self.processed_dir, "test", "fragment", scene_path)
                makedirs(out_dir)
                out_path = osp.join(out_dir, "fragment_{:06d}.pt".format(find_int(f_p)))
                # read ply file
                with open(fragment_path, "rb") as f:
                    data = PlyData.read(f)
                pos = [torch.tensor(data["vertex"][axis]) for axis in ["x", "y", "z"]]
                pos = torch.stack(pos, dim=-1)
                data = Data(pos=pos)
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                torch.save(data, out_path)
                self.table[ind] = {
                    "in_path": fragment_path,
                    "scene_path": scene_path,
                    "fragment_name": f_p,
                    "out_path": out_path,
                }
                ind += 1

        # save this file into json
        with open(osp.join(out_dir, "table.json"), "w") as f:
            json.dump(self.table, f)

    def _compute_matches_between_fragments(self):
        ind = 0
        out_dir = osp.join(self.processed_dir, "test", "matches")
        if files_exist([out_dir]):  # pragma: no cover
            return
        makedirs(out_dir)

        list_scene = os.listdir(osp.join(self.raw_dir, "test"))
        for scene in list_scene:
            path_log = osp.join(self.raw_dir, "test", scene, "gt.log")
            list_pair_num, list_mat = read_gt_log(path_log)
            for i, pair in enumerate(list_pair_num):
                path1 = osp.join(self.processed_dir, "test", "fragment", scene, "fragment_{:06d}.pt".format(pair[0]))
                path2 = osp.join(self.processed_dir, "test", "fragment", scene, "fragment_{:06d}.pt".format(pair[1]))
                data1 = torch.load(path1)
                data2 = torch.load(path2)
                match = compute_overlap_and_matches(
                    data1,
                    data2,
                    self.max_dist_overlap,
                    trans_gt=torch.from_numpy(np.linalg.inv(list_mat[i])).to(data1.pos.dtype),
                )
                match["path_source"] = path1
                match["path_target"] = path2
                match["name_source"] = str(pair[0])
                match["name_target"] = str(pair[1])
                match["scene"] = scene
                out_path = osp.join(self.processed_dir, "test", "matches", "matches{:06d}.npy".format(ind))
                np.save(out_path, match)
                ind += 1

    def process(self):
        self._pre_transform_fragments_ply()
        self._compute_matches_between_fragments()

    def __getitem__(self, idx):
        raise NotImplementedError("implement class to get patch or fragment or more")


class BasePCRBTest(Dataset, GeneralFragment):
    """
    dataset that have the same format as in Point cloud registration benchmark repo.
    https://github.com/iralabdisco/point_clouds_registration_benchmark.
    it
    """


    def __init__(
            self,
            root,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            verbose=False,
            debug=False,
            max_dist_overlap=0.01,
            num_pos_pairs=200,
            self_supervised=False,
            min_points=100,
            min_size_block=2,
            max_size_block=3,
            ss_transform=None,
            use_fps=False,
            is_name_path_int=True,
    ):

        """
        a baseDataset that download a dataset,
        apply preprocessing, and compute keypoints

        Parameters
        ----------
        max_dist_overlap: float
            max distance between points to create a match
        num_pos_pairs: int
            number of positive pairs for the ground truth match
        min_size_block: float
            for self supervised, minimum size of the ball where points inside it will be matched
        max_size_block: float
            for self supervised, maximum size of the ball where points inside it will be matched
        """
        self.is_name_path_int = is_name_path_int
        self.max_dist_overlap = max_dist_overlap
        self.num_pos_pairs = num_pos_pairs
        self.is_online_matching = False
        super(BasePCRBTest, self).__init__(root, transform, pre_transform, pre_filter)
        self.path_match = osp.join(self.processed_dir, "test", "matches")
        self.list_fragment = [f for f in os.listdir(self.path_match) if "matches" in f]
        # Self supervised variables
        self.self_supervised = self_supervised
        self.ss_transform = ss_transform
        self.min_points = min_points
        self.min_size_block = min_size_block
        self.max_size_block = max_size_block
        self.use_fps = use_fps


    def download(self):
        raise NotImplementedError("need to implement the download procedure")

    @staticmethod
    def parse_pair_files(filename):
        with open(filename, "r") as f:
            data = f.readlines()
        res = []
        for i in range(1, len(data)):
            elem = data[i].split(" ")
            trans = [float(t) for t in elem[4:]]
            dico = dict(id=int(elem[0]), source_name=elem[1], target_name=elem[2], overlap=float(elem[3]), trans=trans)
            res.append(dico)
        return res

    @staticmethod
    def read_pcd(filename):
        with open(filename, "r") as f:
            data = f.readlines()
        field = data[2].split("\n")[0].split(" ")[1:]
        num_pt = int(data[9].split("\n")[0].split(" ")[1])
        arr = np.zeros((num_pt, len(field)))
        for i in range(11, len(data)):
            point = data[i].split("\n")[0].split(" ")
            arr[i - 11] = np.array([float(p) for p in point])
        mask = np.prod(~np.isnan(arr), axis=1, dtype=bool)
        arr = arr[mask, :]
        return arr, field

    @property
    def raw_file_names(self):
        return ["test"]

    @property
    def processed_file_names(self):
        res = [osp.join("test", "fragment"), osp.join("test", "matches")]
        return res

    def _pre_transform_fragments(self):
        """
        apply pre_transform on fragments (ply) and save the results
        """
        out_dir = osp.join(self.processed_dir, "test", "fragment")
        if files_exist([out_dir]):  # pragma: no cover
            return
        makedirs(out_dir)

        # table to map fragment numper with
        self.table = dict()
        list_scene = [f for f in os.listdir(osp.join(self.raw_dir, "test"))]
        for scene_path in list_scene:

            pose_path = osp.join(self.raw_dir, "test",
                                 "pose_{}.csv".format(scene_path))

            fragment_dir = osp.join(self.raw_dir, "test", scene_path)


            if osp.isfile(fragment_dir):
                continue
            list_fragment_path = sorted([f for f in os.listdir(fragment_dir) if "pcd" in f])
            for i, f_p in enumerate(list_fragment_path):
                fragment_path = osp.join(fragment_dir, f_p)
                out_dir = osp.join(self.processed_dir, "test", "fragment", scene_path)
                makedirs(out_dir)
                if(self.is_name_path_int):
                    out_path = osp.join(out_dir, "fragment_{:06d}.pt".format(find_int(f_p)))
                else:
                    out_path = osp.join(out_dir, "{}.pt".format(f_p[:-4]))
                pos = torch.from_numpy(BasePCRBTest.read_pcd(fragment_path)[0][:, :3]).float()
                data = Data(pos=pos)
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                if(osp.exists(pose_path)):
                    ind = find_int(f_p)
                    df = pd.read_csv(pose_path)
                    center = torch.tensor(
                        [[df[' T03'][ind], df[' T13'][ind], df[' T23'][ind]]]).float().unsqueeze(0)

                    ind_sensors, _ = dense_knn(data.pos.unsqueeze(0).float(), center, k=1)
                    data.ind_sensors = ind_sensors[0][0]
                else:
                    log.warn("No censors data")

                torch.save(data, out_path)

    def _compute_matches_between_fragments(self):
        ind = 0
        out_dir = osp.join(self.processed_dir, "test", "matches")
        if files_exist([out_dir]):  # pragma: no cover
            return
        makedirs(out_dir)

        list_scene = os.listdir(osp.join(self.raw_dir, "test"))
        for scene in list_scene:
            if osp.isfile(osp.join(self.raw_dir, "test", scene)):
                continue
            path_log = osp.join(self.raw_dir, "test", scene + "_global.txt")
            list_pair = BasePCRBTest.parse_pair_files(path_log)
            for i, pair in enumerate(list_pair):
                if(self.is_name_path_int):
                    name_fragment_s = "fragment_{:06d}.pt".format(find_int(pair["source_name"]))
                    name_fragment_t = "fragment_{:06d}.pt".format(find_int(pair["target_name"]))
                else:
                    name_fragment_s = "{}.pt".format(pair["source_name"][:-4])
                    name_fragment_t = "{}.pt".format(pair["target_name"][:-4])
                path1 = osp.join(
                    self.processed_dir,
                    "test",
                    "fragment",
                    scene,
                    name_fragment_s
                )
                path2 = osp.join(
                    self.processed_dir,
                    "test",
                    "fragment",
                    scene,
                    name_fragment_t
                )
                data1 = torch.load(path1)
                data2 = torch.load(path2)
                match = compute_overlap_and_matches(data1, data2, self.max_dist_overlap)
                match["path_source"] = path1
                match["path_target"] = path2
                match["name_source"] = pair["source_name"]
                match["name_target"] = pair["target_name"]
                match["scene"] = scene
                match["trans"] = pair["trans"]
                out_path = osp.join(self.processed_dir, "test", "matches", "matches{:06d}.npy".format(ind))
                np.save(out_path, match)
                ind += 1

    def process(self):
        self._pre_transform_fragments()
        self._compute_matches_between_fragments()

    def download_pairs(self, path):
        log.info("download pairs")
        req = requests.get(self.link_pairs)
        with open(osp.join(path, "pairs.zip"), "wb") as archive:
            archive.write(req.content)

        with ZipFile(osp.join(path, "pairs.zip"), "r") as zip_obj:
            zip_obj.extractall(path)
        log.info("remove pairs")
        os.remove(osp.join(path, "pairs.zip"))

    def get_raw_pair(self, idx):
        """
        get the pair before the data augmentation
        """
        match = np.load(osp.join(self.path_match, "matches{:06d}.npy".format(idx)), allow_pickle=True).item()

        if not self.self_supervised:
            data_source = torch.load(match["path_source"]).to(torch.float)
            data_target = torch.load(match["path_target"]).to(torch.float)
            new_pair = torch.from_numpy(match["pair"])
            trans = torch.tensor(match["trans"]).reshape(3, 4)
            data_target.pos = data_target.pos @ trans[:3, :3].T + trans[:3, 3]
            if getattr(data_target, "norm", None) is not None:
                data_target.norm = data_target.norm @ trans[:3, :3].T
        else:
            if random.random() < 0.5:
                data_source_o = torch.load(match["path_source"]).to(torch.float)
                data_target_o = torch.load(match["path_source"]).to(torch.float)
            else:
                data_source_o = torch.load(match["path_target"]).to(torch.float)
                data_target_o = torch.load(match["path_target"]).to(torch.float)
            data_source, data_target, new_pair = self.unsupervised_preprocess(data_source_o, data_target_o)
        return data_source, data_target, new_pair

    def __getitem__(self, idx):
        return self.get_fragment(idx)

    def __len__(self):
        return len(self.list_fragment)

    def len(self):
        return len(self)
