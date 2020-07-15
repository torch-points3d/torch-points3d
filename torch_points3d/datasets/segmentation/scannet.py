import os
import os.path as osp
import shutil
import json
import torch
from glob import glob
import sys
import csv
import logging
import numpy as np
from plyfile import PlyData, PlyElement
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
import torch_geometric.transforms as T
import multiprocessing
import pandas as pd

import tempfile
import urllib
from urllib.request import urlopen

from torch_points3d.datasets.base_dataset import BaseDataset
import torch_points3d.core.data_transform as cT
from . import IGNORE_LABEL

log = logging.getLogger(__name__)

# Ref: https://github.com/xjwang-cs/TSDF_utils/blob/master/download-scannet.py
########################################################################################
#                                                                                      #
#                                      Download script                                 #
#                                                                                      #
########################################################################################

BASE_URL = "http://kaldir.vc.in.tum.de/scannet/"
TOS_URL = BASE_URL + "ScanNet_TOS.pdf"
FILETYPES = [
    ".aggregation.json",
    ".sens",
    ".txt",
    "_vh_clean.ply",
    "_vh_clean_2.0.010000.segs.json",
    "_vh_clean_2.ply",
    "_vh_clean.segs.json",
    "_vh_clean.aggregation.json",
    "_vh_clean_2.labels.ply",
    "_2d-instance.zip",
    "_2d-instance-filt.zip",
    "_2d-label.zip",
    "_2d-label-filt.zip",
]
FILETYPES_TEST = [".sens", ".txt", "_vh_clean.ply", "_vh_clean_2.ply"]
PREPROCESSED_FRAMES_FILE = ["scannet_frames_25k.zip", "5.6GB"]
TEST_FRAMES_FILE = ["scannet_frames_test.zip", "610MB"]
LABEL_MAP_FILES = ["scannetv2-labels.combined.tsv", "scannet-labels.combined.tsv"]
RELEASES = ["v2/scans", "v1/scans"]
RELEASES_TASKS = ["v2/tasks", "v1/tasks"]
RELEASES_NAMES = ["v2", "v1"]
RELEASE = RELEASES[0]
RELEASE_TASKS = RELEASES_TASKS[0]
RELEASE_NAME = RELEASES_NAMES[0]
LABEL_MAP_FILE = LABEL_MAP_FILES[0]
RELEASE_SIZE = "1.2TB"
V1_IDX = 1
NUM_CLASSES = 41
CLASS_LABELS = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)
URLS_METADATA = [
    "https://raw.githubusercontent.com/facebookresearch/votenet/master/scannet/meta_data/scannetv2-labels.combined.tsv",
    "https://raw.githubusercontent.com/facebookresearch/votenet/master/scannet/meta_data/scannetv2_train.txt",
    "https://raw.githubusercontent.com/facebookresearch/votenet/master/scannet/meta_data/scannetv2_test.txt",
    "https://raw.githubusercontent.com/facebookresearch/votenet/master/scannet/meta_data/scannetv2_val.txt",
]
VALID_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

SCANNET_COLOR_MAP = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}

SPLITS = ["train", "val", "test"]

MAX_NUM_POINTS = 1200000


def get_release_scans(release_file):
    scan_lines = urlopen(release_file)
    scans = []
    for scan_line in scan_lines:
        scan_id = scan_line.decode("utf8").rstrip("\n")
        scans.append(scan_id)
    return scans


def download_release(release_scans, out_dir, file_types, use_v1_sens):
    if len(release_scans) == 0:
        return
    log.info("Downloading ScanNet " + RELEASE_NAME + " release to " + out_dir + "...")
    failed = []
    for scan_id in release_scans:
        scan_out_dir = os.path.join(out_dir, scan_id)
        try:
            download_scan(scan_id, scan_out_dir, file_types, use_v1_sens)
        except:
            failed.append(scan_id)
    log.info("Downloaded ScanNet " + RELEASE_NAME + " release.")
    if len(failed):
        log.warning("Failed downloads: {}".format(failed))


def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isfile(out_file):
        log.info("\t" + url + " > " + out_file)
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, "w")
        f.close()
        urllib.request.urlretrieve(url, out_file_tmp)
        # urllib.urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)
    else:
        pass
        # log.warning("WARNING Skipping download of existing file " + out_file)


def download_scan(scan_id, out_dir, file_types, use_v1_sens):
    # log.info("Downloading ScanNet " + RELEASE_NAME + " scan " + scan_id + " ...")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for ft in file_types:
        v1_sens = use_v1_sens and ft == ".sens"
        url = (
            BASE_URL + RELEASE + "/" + scan_id + "/" + scan_id + ft
            if not v1_sens
            else BASE_URL + RELEASES[V1_IDX] + "/" + scan_id + "/" + scan_id + ft
        )
        out_file = out_dir + "/" + scan_id + ft
        download_file(url, out_file)
    # log.info("Downloaded scan " + scan_id)


def download_label_map(out_dir):
    log.info("Downloading ScanNet " + RELEASE_NAME + " label mapping file...")
    files = [LABEL_MAP_FILE]
    for file in files:
        url = BASE_URL + RELEASE_TASKS + "/" + file
        localpath = os.path.join(out_dir, file)
        localdir = os.path.dirname(localpath)
        if not os.path.isdir(localdir):
            os.makedirs(localdir)
        download_file(url, localpath)
    log.info("Downloaded ScanNet " + RELEASE_NAME + " label mapping file.")


# REFERENCE TO https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py
########################################################################################
#                                                                                      #
#                                      UTILS                                           #
#                                                                                      #
########################################################################################


def represents_int(s):
    """ if string s represents an int. """
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename, label_from="raw_category", label_to="nyu40id"):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping


def read_mesh_vertices(filename):
    """ read XYZ for each vertex.
    """
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
    return vertices


def read_mesh_vertices_rgb(filename):
    """ read XYZ RGB for each vertex.
    Note: RGB values are in 0-255
    """
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
        vertices[:, 3] = plydata["vertex"].data["red"]
        vertices[:, 4] = plydata["vertex"].data["green"]
        vertices[:, 5] = plydata["vertex"].data["blue"]
    return vertices


def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data["segGroups"])
        for i in range(num_objects):
            object_id = data["segGroups"][i]["objectId"] + 1  # instance ids should be 1-indexed
            label = data["segGroups"][i]["label"]
            segs = data["segGroups"][i]["segments"]
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data["segIndices"])
        for i in range(num_verts):
            seg_id = data["segIndices"][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def export(mesh_file, agg_file, seg_file, meta_file, label_map_file, output_file=None):
    """ points are XYZ RGB (RGB in 0-255),
    semantic label as nyu40 ids,
    instance label as 1-#instance,
    box as (cx,cy,cz,dx,dy,dz,semantic_label)
    """
    label_map = read_label_mapping(label_map_file, label_from="raw_category", label_to="nyu40id")
    mesh_vertices = read_mesh_vertices_rgb(mesh_file)

    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    for line in lines:
        if "axisAlignment" in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    mesh_vertices[:, 0:3] = pts[:, 0:3]

    # Load semantic and instance labels
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]
    instance_bboxes = np.zeros((num_instances, 7))
    for obj_id in object_id_to_segs:
        label_id = object_id_to_label_id[obj_id]
        obj_pc = mesh_vertices[instance_ids == obj_id, 0:3]
        if len(obj_pc) == 0:
            continue
        # Compute axis aligned box
        # An axis aligned bounding box is parameterized by
        # (cx,cy,cz) and (dx,dy,dz) and label id
        # where (cx,cy,cz) is the center point of the box,
        # dx is the x-axis length of the box.
        xmin = np.min(obj_pc[:, 0])
        ymin = np.min(obj_pc[:, 1])
        zmin = np.min(obj_pc[:, 2])
        xmax = np.max(obj_pc[:, 0])
        ymax = np.max(obj_pc[:, 1])
        zmax = np.max(obj_pc[:, 2])
        bbox = np.array(
            [
                (xmin + xmax) / 2.0,
                (ymin + ymax) / 2.0,
                (zmin + zmax) / 2.0,
                xmax - xmin,
                ymax - ymin,
                zmax - zmin,
                label_id,
            ]
        )
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        instance_bboxes[obj_id - 1, :] = bbox

    return (
        mesh_vertices.astype(np.float32),
        label_ids.astype(np.int),
        instance_ids.astype(np.int),
        instance_bboxes.astype(np.float32),
        object_id_to_label_id,
    )


########################################################################################
#                                                                                      #
#                          SCANNET InMemoryDataset DATASET                             #
#                                                                                      #
########################################################################################


class Scannet(InMemoryDataset):
    """ Scannet dataset, you will have to agree to terms and conditions by hitting enter
    so that it downloads the dataset.

    http://www.scan-net.org/

    Parameters
    ----------
    root : str
        Path to the data
    split : str, optional
        Split used (train, val or test)
    transform (callable, optional):
        A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
    pre_transform (callable, optional): 
        A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a 
        transformed version. The data object will be transformed before being saved to disk.
    pre_filter (callable, optional): 
        A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the final dataset. 
    version : str, optional
        version of scannet, by default "v2"
    use_instance_labels : bool, optional
        Wether we use instance labels or not, by default False
    use_instance_bboxes : bool, optional
        Wether we use bounding box labels or not, by default False
    donotcare_class_ids : list, optional
        Class ids to be discarded
    max_num_point : [type], optional
        Max number of points to keep during the pre processing step
    use_multiprocessing : bool, optional
        Wether we use multiprocessing or not
    process_workers : int, optional
        Number of process workers
    normalize_rgb : bool, optional
        Normalise rgb values, by default True
    """

    CLASS_LABELS = CLASS_LABELS
    URLS_METADATA = URLS_METADATA
    VALID_CLASS_IDS = VALID_CLASS_IDS
    SCANNET_COLOR_MAP = SCANNET_COLOR_MAP
    SPLITS = SPLITS

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        version="v2",
        use_instance_labels=False,
        use_instance_bboxes=False,
        donotcare_class_ids=[],
        max_num_point=None,
        process_workers=4,
        types=[".txt", "_vh_clean_2.ply", "_vh_clean_2.0.010000.segs.json", ".aggregation.json"],
        normalize_rgb=True,
        is_test=False,
    ):

        assert self.SPLITS == ["train", "val", "test"]
        if not isinstance(donotcare_class_ids, list):
            raise Exception("donotcare_class_ids should be list with indices of class to ignore")
        self.donotcare_class_ids = donotcare_class_ids

        self.valid_class_idx = [idx for idx in self.VALID_CLASS_IDS if idx not in donotcare_class_ids]

        assert version in ["v2", "v1"], "The version should be either v1 or v2"
        self.version = version
        self.max_num_point = max_num_point
        self.use_instance_labels = use_instance_labels
        self.use_instance_bboxes = use_instance_bboxes
        self.use_multiprocessing = process_workers > 1
        self.process_workers = process_workers
        self.types = types
        self.normalize_rgb = normalize_rgb
        self.is_test = is_test
        super().__init__(root, transform, pre_transform, pre_filter)
        if split == "train":
            path = self.processed_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
        else:
            raise ValueError((f"Split {split} found, but expected either " "train, val, or test"))

        self.data, self.slices = torch.load(path)

        if split != "test":
            if not use_instance_bboxes:
                delattr(self.data, "instance_bboxes")
            if not use_instance_labels:
                delattr(self.data, "instance_labels")
            self.data.y = self._remap_labels(self.data.y)
            self.has_labels = True
        else:
            self.has_labels = False

        self.read_from_metadata()

    def get_raw_data(self, id_scan, remap_labels=True) -> Data:
        """ Grabs the raw data associated with a scan index

        Parameters
        ----------
        id_scan : int or torch.Tensor
            id of the scan
        remap_labels : bool, optional
            If True then labels are mapped to the range [IGNORE_LABELS:number of labels]. 
            If using this method to compare ground truth against prediction then set remap_labels to True
        """
        stage = self.name
        if torch.is_tensor(id_scan):
            id_scan = int(id_scan.item())
        assert stage in self.SPLITS
        mapping_idx_to_scan_names = getattr(self, "MAPPING_IDX_TO_SCAN_{}_NAMES".format(stage.upper()))
        scan_name = mapping_idx_to_scan_names[id_scan]
        path_to_raw_scan = os.path.join(
            self.processed_raw_paths[self.SPLITS.index(stage.lower())], "{}.pt".format(scan_name)
        )
        data = torch.load(path_to_raw_scan)
        data.scan_name = scan_name
        data.path_to_raw_scan = path_to_raw_scan
        if self.has_labels and remap_labels:
            data.y = self._remap_labels(data.y)
        return data

    @property
    def raw_file_names(self):
        return ["metadata", "scans", "scannetv2-labels.combined.tsv"]

    @property
    def processed_file_names(self):
        return ["{}.pt".format(s,) for s in Scannet.SPLITS]

    @property
    def processed_raw_paths(self):
        processed_raw_paths = [os.path.join(self.processed_dir, "raw_{}".format(s)) for s in Scannet.SPLITS]
        for p in processed_raw_paths:
            if not os.path.exists(p):
                os.makedirs(p)
        return processed_raw_paths

    @property
    def path_to_submission(self):
        root = os.getcwd()
        path_to_submission = os.path.join(root, "submission_labels")
        if not os.path.exists(path_to_submission):
            os.makedirs(path_to_submission)
        return path_to_submission

    @property
    def num_classes(self):
        return len(Scannet.VALID_CLASS_IDS)

    def download_scans(self):
        release_file = BASE_URL + RELEASE + ".txt"
        release_scans = get_release_scans(release_file)
        # release_scans = ["scene0191_00","scene0191_01", "scene0568_00", "scene0568_01"]
        file_types = FILETYPES
        release_test_file = BASE_URL + RELEASE + "_test.txt"
        release_test_scans = get_release_scans(release_test_file)
        file_types_test = FILETYPES_TEST
        out_dir_scans = os.path.join(self.raw_dir, "scans")
        out_dir_test_scans = os.path.join(self.raw_dir, "scans_test")

        if self.types:  # download file type
            file_types = self.types
            for file_type in file_types:
                if file_type not in FILETYPES:
                    log.error("ERROR: Invalid file type: " + file_type)
                    return
            file_types_test = []
            for file_type in file_types:
                if file_type in FILETYPES_TEST:
                    file_types_test.append(file_type)
        download_label_map(self.raw_dir)
        log.info("WARNING: You are downloading all ScanNet " + RELEASE_NAME + " scans of type " + file_types[0])
        log.info(
            "Note that existing scan directories will be skipped. Delete partially downloaded directories to re-download."
        )
        log.info("***")
        log.info("Press any key to continue, or CTRL-C to exit.")
        input("")
        if self.version == "v2" and ".sens" in file_types:
            log.info(
                "Note: ScanNet v2 uses the same .sens files as ScanNet v1: Press 'n' to exclude downloading .sens files for each scan"
            )
            key = input("")
            if key.strip().lower() == "n":
                file_types.remove(".sens")
        download_release(release_scans, out_dir_scans, file_types, use_v1_sens=True)
        if self.version == "v2":
            download_label_map(self.raw_dir)
            download_release(release_test_scans, out_dir_test_scans, file_types_test, use_v1_sens=True)
            # download_file(os.path.join(BASE_URL, RELEASE_TASKS, TEST_FRAMES_FILE[0]), os.path.join(out_dir_tasks, TEST_FRAMES_FILE[0]))

    def download(self):
        if self.is_test:
            return
        log.info(
            "By pressing any key to continue you confirm that you have agreed to the ScanNet terms of use as described at:"
        )
        log.info(TOS_URL)
        log.info("***")
        log.info("Press any key to continue, or CTRL-C to exit.")
        input("")
        self.download_scans()
        metadata_path = osp.join(self.raw_dir, "metadata")
        if not os.path.exists(metadata_path):
            os.makedirs(metadata_path)
        for url in self.URLS_METADATA:
            _ = download_url(url, metadata_path)

    @staticmethod
    def read_one_test_scan(scannet_dir, scan_name, normalize_rgb):
        mesh_file = osp.join(scannet_dir, scan_name, scan_name + "_vh_clean_2.ply")
        mesh_vertices = read_mesh_vertices_rgb(mesh_file)

        data = {}
        data["pos"] = torch.from_numpy(mesh_vertices[:, :3])
        data["rgb"] = torch.from_numpy(mesh_vertices[:, 3:])
        if normalize_rgb:
            data["rgb"] /= 255.0
        return Data(**data)

    @staticmethod
    def read_one_scan(
        scannet_dir, scan_name, label_map_file, donotcare_class_ids, max_num_point, obj_class_ids, normalize_rgb,
    ):
        mesh_file = osp.join(scannet_dir, scan_name, scan_name + "_vh_clean_2.ply")
        agg_file = osp.join(scannet_dir, scan_name, scan_name + ".aggregation.json")
        seg_file = osp.join(scannet_dir, scan_name, scan_name + "_vh_clean_2.0.010000.segs.json")
        meta_file = osp.join(
            scannet_dir, scan_name, scan_name + ".txt"
        )  # includes axisAlignment info for the train set scans.
        mesh_vertices, semantic_labels, instance_labels, instance_bboxes, instance2semantic = export(
            mesh_file, agg_file, seg_file, meta_file, label_map_file, None
        )

        # Discard unwanted classes
        mask = np.logical_not(np.in1d(semantic_labels, donotcare_class_ids))
        mesh_vertices = mesh_vertices[mask, :]
        semantic_labels = semantic_labels[mask]
        instance_labels = instance_labels[mask]

        bbox_mask = np.in1d(instance_bboxes[:, -1], obj_class_ids)
        instance_bboxes = instance_bboxes[bbox_mask, :]

        # Subsample
        N = mesh_vertices.shape[0]
        if max_num_point:
            if N > max_num_point:
                choices = np.random.choice(N, max_num_point, replace=False)
                mesh_vertices = mesh_vertices[choices, :]
                semantic_labels = semantic_labels[choices]
                instance_labels = instance_labels[choices]

        # Build data container
        data = {}
        data["pos"] = torch.from_numpy(mesh_vertices[:, :3])
        data["rgb"] = torch.from_numpy(mesh_vertices[:, 3:])
        if normalize_rgb:
            data["rgb"] /= 255.0
        data["y"] = torch.from_numpy(semantic_labels)
        data["x"] = None
        data["instance_labels"] = torch.from_numpy(instance_labels)
        data["instance_bboxes"] = torch.from_numpy(instance_bboxes)

        return Data(**data)

    def read_from_metadata(self):
        metadata_path = osp.join(self.raw_dir, "metadata")
        self.label_map_file = osp.join(metadata_path, LABEL_MAP_FILE)
        split_files = ["scannetv2_{}.txt".format(s) for s in Scannet.SPLITS]
        self.scan_names = []
        for sf in split_files:
            f = open(osp.join(metadata_path, sf))
            self.scan_names.append(sorted([line.rstrip() for line in f]))
            f.close()

        for idx_split, split in enumerate(Scannet.SPLITS):
            idx_mapping = {idx: scan_name for idx, scan_name in enumerate(self.scan_names[idx_split])}
            setattr(self, "MAPPING_IDX_TO_SCAN_{}_NAMES".format(split.upper()), idx_mapping)

    @staticmethod
    def process_func(
        id_scan,
        total,
        scannet_dir,
        scan_name,
        label_map_file,
        donotcare_class_ids,
        max_num_point,
        obj_class_ids,
        normalize_rgb,
        split,
    ):
        if split == "test":
            data = Scannet.read_one_test_scan(scannet_dir, scan_name, normalize_rgb)
        else:
            data = Scannet.read_one_scan(
                scannet_dir,
                scan_name,
                label_map_file,
                donotcare_class_ids,
                max_num_point,
                obj_class_ids,
                normalize_rgb,
            )
        log.info("{}/{}| scan_name: {}, data: {}".format(id_scan, total, scan_name, data))

        data["id_scan"] = torch.tensor([id_scan])
        return cT.SaveOriginalPosId()(data)

    def process(self):
        if self.is_test:
            return
        self.read_from_metadata()

        scannet_dir = osp.join(self.raw_dir, "scans")
        for i, (scan_names, split) in enumerate(zip(self.scan_names, self.SPLITS)):
            if not os.path.exists(self.processed_paths[i]):
                mapping_idx_to_scan_names = getattr(self, "MAPPING_IDX_TO_SCAN_{}_NAMES".format(split.upper()))
                scannet_dir = osp.join(self.raw_dir, "scans" if split in ["train", "val"] else "scans_test")
                total = len(scan_names)
                args = [
                    (
                        id,
                        total,
                        scannet_dir,
                        scan_name,
                        self.label_map_file,
                        self.donotcare_class_ids,
                        self.max_num_point,
                        self.VALID_CLASS_IDS,
                        self.normalize_rgb,
                        split,
                    )
                    for id, scan_name in enumerate(scan_names)
                ]
                if self.use_multiprocessing:
                    with multiprocessing.Pool(processes=self.process_workers) as pool:
                        datas = pool.starmap(Scannet.process_func, args)
                else:
                    datas = []
                    for arg in args:
                        data = Scannet.process_func(*arg)
                        datas.append(data)

                for data in datas:
                    id_scan = int(data.id_scan.item())
                    scan_name = mapping_idx_to_scan_names[id_scan]
                    path_to_raw_scan = os.path.join(self.processed_raw_paths[i], "{}.pt".format(scan_name))
                    torch.save(data, path_to_raw_scan)

                datas = [self.pre_transform(data) for data in datas]

                log.info("SAVING TO {}".format(self.processed_paths[i]))
                torch.save(self.collate(datas), self.processed_paths[i])

    def _remap_labels(self, semantic_label):
        """ Remaps labels to [0 ; num_labels -1]. Can be overriden.
        """
        new_labels = semantic_label.clone()
        mapping_dict = {indice: idx for idx, indice in enumerate(self.valid_class_idx)}
        for idx in range(NUM_CLASSES):
            if idx not in mapping_dict:
                mapping_dict[idx] = IGNORE_LABEL
        for idx in self.donotcare_class_ids:
            mapping_dict[idx] = IGNORE_LABEL
        for source, target in mapping_dict.items():
            mask = semantic_label == source
            new_labels[mask] = target

        return new_labels

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, len(self))


class ScannetDataset(BaseDataset):
    """ Wrapper around Scannet that creates train and test datasets.

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - version
            - max_num_point (optional)
            - use_instance_labels (optional)
            - use_instance_bboxes (optional)
            - donotcare_class_ids (optional)
            - pre_transforms (optional)
            - train_transforms (optional)
            - val_transforms (optional)
    """

    SPLITS = SPLITS

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        use_instance_labels: bool = dataset_opt.use_instance_labels
        use_instance_bboxes: bool = dataset_opt.use_instance_bboxes
        donotcare_class_ids: [] = dataset_opt.donotcare_class_ids if dataset_opt.donotcare_class_ids else []
        max_num_point: int = dataset_opt.max_num_point if dataset_opt.max_num_point is not None else None
        process_workers: int = dataset_opt.process_workers if dataset_opt.process_workers else 0
        is_test: bool = dataset_opt.is_test if dataset_opt.is_test is not None else False

        self.train_dataset = Scannet(
            self._data_path,
            split="train",
            pre_transform=self.pre_transform,
            transform=self.train_transform,
            version=dataset_opt.version,
            use_instance_labels=use_instance_labels,
            use_instance_bboxes=use_instance_bboxes,
            donotcare_class_ids=donotcare_class_ids,
            max_num_point=max_num_point,
            process_workers=process_workers,
            is_test=is_test,
        )

        self.val_dataset = Scannet(
            self._data_path,
            split="val",
            transform=self.val_transform,
            pre_transform=self.pre_transform,
            version=dataset_opt.version,
            use_instance_labels=use_instance_labels,
            use_instance_bboxes=use_instance_bboxes,
            donotcare_class_ids=donotcare_class_ids,
            max_num_point=max_num_point,
            process_workers=process_workers,
            is_test=is_test,
        )

        self.test_dataset = Scannet(
            self._data_path,
            split="test",
            transform=self.val_transform,
            pre_transform=self.pre_transform,
            version=dataset_opt.version,
            use_instance_labels=use_instance_labels,
            use_instance_bboxes=use_instance_bboxes,
            donotcare_class_ids=donotcare_class_ids,
            max_num_point=max_num_point,
            process_workers=process_workers,
            is_test=is_test,
        )

    @property
    def path_to_submission(self):
        return self.train_dataset.path_to_submission

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            dataset {[type]}
            wandb_log - Log using weight and biases
        Returns:
            [BaseTracker] -- tracker
        """
        from torch_points3d.metrics.scannet_segmentation_tracker import ScannetSegmentationTracker

        return ScannetSegmentationTracker(
            self, wandb_log=wandb_log, use_tensorboard=tensorboard_log, ignore_label=IGNORE_LABEL
        )
