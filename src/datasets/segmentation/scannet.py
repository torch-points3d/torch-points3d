import os
import os.path as osp
import shutil
import json
from tqdm import tqdm as tq
import torch
from glob import glob
import sys
import json
import csv

from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_txt_array
import torch_geometric.transforms as T

from src.metrics.segmentation_tracker import SegmentationTracker

from src.datasets.base_dataset import BaseDataset


# REFERENCE TO https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py
###################### UTILS ##################
try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)

def represents_int(s):
    ''' if string s represents an int. '''
    try: 
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping

def read_mesh_vertices(filename):
    """ read XYZ for each vertex.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
    return vertices

def read_mesh_vertices_rgb(filename):
    """ read XYZ RGB for each vertex.
    Note: RGB values are in 0-255
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
    return vertices


def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
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
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
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
    label_map = read_label_mapping(label_map_file,
        label_from='raw_category', label_to='nyu40id')    
    mesh_vertices = read_mesh_vertices_rgb(mesh_file)

    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:,0:3] = mesh_vertices[:,0:3]
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    mesh_vertices[:,0:3] = pts[:,0:3]

    # Load semantic and instance labels
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]
    instance_bboxes = np.zeros((num_instances,7))
    for obj_id in object_id_to_segs:
        label_id = object_id_to_label_id[obj_id]
        obj_pc = mesh_vertices[instance_ids==obj_id, 0:3]
        if len(obj_pc) == 0: continue
        # Compute axis aligned box
        # An axis aligned bounding box is parameterized by
        # (cx,cy,cz) and (dx,dy,dz) and label id
        # where (cx,cy,cz) is the center point of the box,
        # dx is the x-axis length of the box.
        xmin = np.min(obj_pc[:,0])
        ymin = np.min(obj_pc[:,1])
        zmin = np.min(obj_pc[:,2])
        xmax = np.max(obj_pc[:,0])
        ymax = np.max(obj_pc[:,1])
        zmax = np.max(obj_pc[:,2])
        bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2,
            xmax-xmin, ymax-ymin, zmax-zmin, label_id])
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        instance_bboxes[obj_id-1,:] = bbox 

    return mesh_vertices.astype(np.float32), label_ids.astype(np.int), instance_ids.astype(np.int),\
        instance_bboxes.astype(np.float32), object_id_to_label_id

######################## SCANNET DATASET ##########################

class Scannet(InMemoryDataset):
    CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                    'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                    'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
    URLS_METADATA = ["https://raw.githubusercontent.com/facebookresearch/votenet/master/scannet/meta_data/scannetv2-labels.combined.tsv", 
                     "https://raw.githubusercontent.com/facebookresearch/votenet/master/scannet/meta_data/scannetv2_train.txt",
                     "https://raw.githubusercontent.com/facebookresearch/votenet/master/scannet/meta_data/scannetv2_test.txt",
                     "https://raw.githubusercontent.com/facebookresearch/votenet/master/scannet/meta_data/scannetv2_val.txt"]
    VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
    SCANNET_COLOR_MAP = {
        0: (0., 0., 0.),
        1: (174., 199., 232.),
        2: (152., 223., 138.),
        3: (31., 119., 180.),
        4: (255., 187., 120.),
        5: (188., 189., 34.),
        6: (140., 86., 75.),
        7: (255., 152., 150.),
        8: (214., 39., 40.),
        9: (197., 176., 213.),
        10: (148., 103., 189.),
        11: (196., 156., 148.),
        12: (23., 190., 207.),
        14: (247., 182., 210.),
        15: (66., 188., 102.),
        16: (219., 219., 141.),
        17: (140., 57., 197.),
        18: (202., 185., 52.),
        19: (51., 176., 203.),
        20: (200., 54., 131.),
        21: (92., 193., 61.),
        22: (78., 71., 183.),
        23: (172., 114., 82.),
        24: (255., 127., 14.),
        25: (91., 163., 138.),
        26: (153., 98., 156.),
        27: (140., 153., 101.),
        28: (158., 218., 229.),
        29: (100., 125., 154.),
        30: (178., 127., 135.),
        32: (146., 111., 194.),
        33: (44., 160., 44.),
        34: (112., 128., 144.),
        35: (96., 207., 209.),
        36: (227., 119., 194.),
        37: (213., 92., 176.),
        38: (94., 106., 211.),
        39: (82., 84., 163.),
        40: (100., 85., 144.),
    }

    SPLIT = ["train", "val", "test"]

    DONOTCARE_CLASS_IDS = np.array([])

    def __init__(
        self,
        root,
        split="trainval",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        version="v2",
        use_instance_labels=False,
        use_instance_bboxes=False,
        donotcare_class_ids=None,
        max_num_point=None
    ):

        if donotcare_class_ids:
            self.DONOTCARE_CLASS_IDS = np.asarray(donotcare_class_ids)
        assert version in ["v2", "v1"], "The version should be either v1 or v2"
        self.version = version
        self.max_num_point = max_num_point
        self.use_instance_labels = use_instance_labels
        self.use_instance_bboxes = use_instance_bboxes

        super(Scannet, self).__init__(root, transform, pre_transform, pre_filter)

        if split == "train":
            path = self.processed_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
        else:
            raise ValueError((f"Split {split} found, but expected either " "train, val, trainval or test"))

        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return glob(osp.join(self.raw_dir, "scans", "*"))

    @property
    def processed_file_names(self):
        return ["{}.pt".format(s,) for s in self.SPLIT]

    def download(self):
        if len(self.raw_file_names) == 0:
            raise Exception("Please, run poetry run python scripts/datasets/download-scannet.py -o data/scannet/raw/ --types _vh_clean.segs.json --types .txt  _vh_clean_2.ply _vh_clean_2.0.010000.segs.json .aggregation.json")
        else:
            metadata_path = osp.join(self.raw_dir, "metadata")
            if not os.path.exists(metadata_path):
                os.makedirs(metadata_path)
            for url in self.URLS_METADATA:
                _ = download_url(url, metadata_path)

    @staticmethod
    def read_one_scan(scannet_dir, scan_name, label_map_file, donotcare_class_ids, max_num_point, obj_class_ids, use_instance_labels=True, use_instance_bboxes=True):    
        mesh_file = osp.join(scannet_dir, scan_name, scan_name + '_vh_clean_2.ply')
        agg_file = osp.join(scannet_dir, scan_name, scan_name + '.aggregation.json')
        seg_file = osp.join(scannet_dir, scan_name, scan_name + '_vh_clean_2.0.010000.segs.json')
        meta_file = osp.join(scannet_dir, scan_name, scan_name + '.txt') # includes axisAlignment info for the train set scans.   
        mesh_vertices, semantic_labels, instance_labels, instance_bboxes, instance2semantic = \
            export(mesh_file, agg_file, seg_file, meta_file, label_map_file, None)

        mask = np.logical_not(np.in1d(semantic_labels, donotcare_class_ids))
        mesh_vertices = mesh_vertices[mask,:]
        semantic_labels = semantic_labels[mask]
        instance_labels = instance_labels[mask]

        num_instances = len(np.unique(instance_labels))
        print('Num of instances: ', num_instances)

        bbox_mask = np.in1d(instance_bboxes[:,-1], obj_class_ids)
        instance_bboxes = instance_bboxes[bbox_mask,:]
        print('Num of care instances: ', instance_bboxes.shape[0])

        N = mesh_vertices.shape[0]
        if max_num_point:
            if N > max_num_point:
                choices = np.random.choice(N, max_num_point, replace=False)
                mesh_vertices = mesh_vertices[choices, :]
                semantic_labels = semantic_labels[choices]
                instance_labels = instance_labels[choices]

        data = {}
        data["pos"] = torch.from_numpy(mesh_vertices[:, :3])
        data["rgb"] = torch.from_numpy(mesh_vertices[:, 3:]) / 255.
        data["y"] = torch.from_numpy(semantic_labels)
        
        if use_instance_labels:
            data["iy"] = torch.from_numpy(instance_labels)
        
        if use_instance_bboxes:
            data["bbox"] = torch.from_numpy(instance_bboxes)

        return Data(**data)

    def read_from_metadata(self):
        metadata_path = osp.join(self.raw_dir, "metadata")
        self.LABEL_MAP_FILE = osp.join(metadata_path, "scannetv2-labels.combined.tsv")
        split_files = ["scannetv2_{}.txt".format(s) for s in self.SPLIT]
        self.SCAN_NAMES = [[line.rstrip() for line in open(osp.join(metadata_path, sf))]for sf in split_files]

    def process(self):
        self.download()
        self.read_from_metadata()

        scannet_dir = osp.join(self.raw_dir, "scans")

        for i, (scan_names, split) in enumerate(zip(self.SCAN_NAMES, self.SPLIT)):
            datas = []

            for scan_name in scan_names:
                print("scan_name: {}".format(scan_name))
                data = Scannet.read_one_scan(scannet_dir, scan_name, self.LABEL_MAP_FILE, self.DONOTCARE_CLASS_IDS, 
                                             self.max_num_point, self.VALID_CLASS_IDS, self.use_instance_labels, self.use_instance_bboxes)

                if self.pre_transform:
                    data = self.pre_transform(data)

                datas.append(data)

            torch.save(self.collate(datas), self.processed_paths[i])


    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, len(self))


class ScannetDataset(BaseDataset):

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        self.train_dataset = Scannet(
            self._data_path,
            split="train",
            pre_transform=self.pre_transform,
            transform=self.train_transform,
            version=dataset_opt.version
        )

        self.val_dataset = Scannet(
            self._data_path,
            split="val",
            transform=self.test_transform,
            pre_transform=self.pre_transform,
            version=dataset_opt.version
        )

        self.test_dataset = Scannet(
            self._data_path,
            split="test",
            transform=self.test_transform,
            pre_transform=self.pre_transform,
            version=dataset_opt.version
        )

    @staticmethod
    def get_tracker(model, dataset, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            dataset {[type]}
            wandb_log - Log using weight and biases
        Returns:
            [BaseTracker] -- tracker
        """
        return SegmentationTracker(dataset, wandb_log=wandb_log, use_tensorboard=tensorboard_log)

