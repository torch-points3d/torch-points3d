import glob
import json
import logging
import numpy as np
import os
import os.path as osp

import shutil
import torch

from torch_geometric.data import Dataset, download_url, extract_zip
from torch_geometric.data import Data
from torch_points3d.datasets.registration.detector import RandomDetector
from torch_points3d.datasets.registration.utils import rgbd2fragment_rough
from torch_points3d.datasets.registration.utils import rgbd2fragment_fine
from torch_points3d.datasets.registration.utils import compute_overlap_and_matches
from torch_points3d.datasets.registration.utils import to_list
from torch_points3d.datasets.registration.utils import files_exist
from torch_points3d.datasets.registration.utils import makedirs
from torch_points3d.datasets.registration.utils import get_urls
from torch_points3d.datasets.registration.utils import PatchExtractor

log = logging.getLogger(__name__)


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    filedata = {}

    with open(filepath, "r") as f:
        for line in f.readlines():
            key, value = line.split(":", 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                filedata[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    T_calib = np.reshape(filedata["Tr"], (3, 4))
    T_calib = np.vstack([T_calib, [0, 0, 0, 1]])

    return T_calib


def compute_spaced_time_frame(list_name_frames, path, min_dist):
    """
    compute spaced time frame to have different pairs.
    Inspired by
    """
    list_pair = []
    list_sensor_pos = []
    for name in list_name_frames:
        data = torch.load(osp.join(path, name))
        list_sensor_pos.append(data.pos_sensor)

    list_sensor_pos = torch.stack(list_sensor_pos)

    mask = ((list_sensor_pos.unsqueeze(0) - list_sensor_pos.unsqueeze(1)) ** 2).sum(2) > (min_dist ** 2)
    curr_ind = 0

    for curr_ind in range(len(list_name_frames)):
        candidates = torch.where(mask[curr_ind, curr_ind : curr_ind + 100])[0]
        if len(candidates) > 0:
            new_ind = curr_ind + candidates[0].item()
            list_pair.append((curr_ind, new_ind))
        else:
            continue
    return list_pair


class BaseKitti(Dataset):
    sequence_train = [0, 1, 2, 3, 4, 5]
    sequence_val = [6, 7]
    sequence_test = [8, 9, 10]
    dict_seq = dict(train=sequence_train, val=sequence_val, test=sequence_test)

    def __init__(
        self,
        root,
        mode="train",
        max_dist_overlap=0.01,
        max_time_distance=3,
        min_dist=10,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):

        """
        KITTI Odometry dataset for pair registration
        """

        self.mode = mode
        self.max_dist_overlap = max_dist_overlap
        self.max_time_distance = max_time_distance
        self.min_dist = min_dist
        if mode not in self.dict_seq.keys():
            raise RuntimeError("this mode {} does " "not exist" "(train|val|test)".format(mode))
        super(BaseKitti, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [osp.join("dataset", "poses"), osp.join("dataset", "refined_poses"), osp.join("dataset", "sequences")]

    @property
    def processed_file_names(self):
        res = [osp.join(self.mode, "matches"), osp.join(self.mode, "fragment")]
        return res

    def download(self):
        log.info("WARNING: You need to first download the kitti dataset (velodyne laser data) on this website")
        log.info("http://www.cvlibs.net/datasets/kitti/eval_odometry.php")
        log.info("you also need to download the refined pose here")
        log.info("https://cloud.mines-paristech.fr/index.php/s/1t1CdXxv4i2v1zC")
        log.info("WARNING: ")
        log.info("the tree should look like this:")
        log.info(
            """
        raw
        ├── dataset
        │   ├── refined_poses
        │   └── sequences
        │       ├── 00
        │       │   └── velodyne
        │       ├── 01
        │       │   └── velodyne
        │       ├── 02
        │       │   └── velodyne
        │       ├── 03
        │       │   └── velodyne
        │       ├── 04
        │       │   └── velodyne
        │       ├── 05
        │       │   └── velodyne
        │       ├── 06
        │       │   └── velodyne
        │       ├── 07
        │       │   └── velodyne
        │       ├── 08
        │       │   └── velodyne
        │       ├── 09
        │       │   └── velodyne
        │       ├── 10
        │       │   └── velodyne

        """
        )

    def _pre_transform_fragment(self, mod):
        """
        read raw fragment, rotate the raw fragment using the calibration and the given pose,
        pre transform raw_fragment and save it into fragments
        """

        if files_exist([osp.join(self.processed_dir, mod, "fragment")]):  # pragma: no cover
            return

        in_dir = osp.join(self.raw_dir, "dataset")
        list_drive = self.dict_seq[mod]

        for drive in list_drive:
            out_dir = osp.join(self.processed_dir, mod, "fragment", "{:02d}".format(drive))
            makedirs(out_dir)
            path_frames = osp.join(in_dir, "sequences", "{:02d}".format(drive), "velodyne")
            T_calib = read_calib_file(osp.join(in_dir, "sequences", "{:02d}".format(drive), "calib.txt"))
            all_poses = np.genfromtxt(osp.join(in_dir, "refined_poses", "{:02d}.txt".format(drive)))
            list_name_frames = sorted([f for f in os.listdir(path_frames) if "bin" in f])
            for i, name in enumerate(list_name_frames):

                pose = all_poses[i].reshape((3, 4))
                xyzr = np.fromfile(osp.join(path_frames, name), dtype=np.float32).reshape((-1, 4))
                xyzr[:, :3] = xyzr[:, :3].dot(T_calib[:3, :3].T) + T_calib[:3, 3]
                xyzr[:, :3] = xyzr[:, :3].dot(pose[:3, :3].T) + pose[:3, 3]
                # store position of the car to filter some frames
                pos_sensor = pose[:3, :3].dot(T_calib[:3, 3]) + pose[:3, 3]
                data = Data(
                    pos=torch.from_numpy(xyzr[:, :3]),
                    reflectance=torch.from_numpy(xyzr[:, 3]),
                    pos_sensor=torch.from_numpy(pos_sensor),
                )
                out_path = osp.join(out_dir, name.split(".")[0] + ".pt")
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                torch.save(data, out_path)

    def _compute_matches_between_fragments(self, mod):
        out_dir = osp.join(self.processed_dir, mod, "matches")
        if files_exist([out_dir]):  # pragma: no cover
            return
        makedirs(out_dir)
        ind = 0
        list_drive = self.dict_seq[mod]
        for drive in list_drive:
            path_fragment = osp.join(self.processed_dir, mod, "fragment", "{:02d}".format(drive))
            list_name_frames = sorted([f for f in os.listdir(path_fragment) if "pt" in f])

            # pre_compute specific pair
            log.info("Compute the pairs")
            if self.min_dist is not None:
                pair_time_frame = compute_spaced_time_frame(list_name_frames, path_fragment, self.min_dist)
            else:
                pair_time_frame = [
                    (i, j)
                    for i in range(len(list_name_frames))
                    for j in range(len(list_name_frames))
                    if (j - i) > 0 and (j - i) < self.max_time_distance
                ]
            log.info("Compute the matches")
            for i, j in pair_time_frame:
                out_path = osp.join(out_dir, "matches{:06d}.npy".format(ind))
                path1 = osp.join(path_fragment, list_name_frames[i])
                path2 = osp.join(path_fragment, list_name_frames[j])
                data1 = torch.load(path1)
                data2 = torch.load(path2)
                match = compute_overlap_and_matches(data1, data2, self.max_dist_overlap)
                match["path_source"] = path1
                match["path_target"] = path2
                match["name_source"] = i
                match["name_target"] = j
                match["scene"] = drive
                np.save(out_path, match)
                ind += 1

    def process(self):
        log.info("pre_transform those fragments")
        self._pre_transform_fragment(self.mode)
        log.info("compute matches")
        self._compute_matches_between_fragments(self.mode)

    def get(self, idx):
        raise NotImplementedError("implement class to get patch or fragment or more")

    def __getitem__(self, idx):
        r"""Gets the data object at index :obj:`idx` and transforms it (in case
        a :obj:`self.transform` is given).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, a  LongTensor or a BoolTensor, will return a subset of the
        dataset at the specified indices."""

        data = self.get(idx)
        return data
