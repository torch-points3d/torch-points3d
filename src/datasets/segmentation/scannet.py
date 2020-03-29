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
import logging
import numpy as np
from plyfile import PlyData, PlyElement
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_txt_array
import torch_geometric.transforms as T
import multiprocessing
from tqdm import tqdm

import tempfile
import urllib

if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    from urllib import urlopen

from src.metrics.scannet_tracker import ScannetTracker
from src.datasets.base_dataset import BaseDataset

log = logging.getLogger(__name__)

# Ref: https://github.com/xjwang-cs/TSDF_utils/blob/master/download-scannet.py
########################################################################################
#                                                                                      #
#                                      Download script                                 #
#                                                                                      #
########################################################################################

BASE_URL = 'http://kaldir.vc.in.tum.de/scannet/'
TOS_URL = BASE_URL + 'ScanNet_TOS.pdf'
FILETYPES = ['.aggregation.json', '.sens', '.txt', '_vh_clean.ply', '_vh_clean_2.0.010000.segs.json', '_vh_clean_2.ply', '_vh_clean.segs.json', '_vh_clean.aggregation.json', '_vh_clean_2.labels.ply', '_2d-instance.zip', '_2d-instance-filt.zip', '_2d-label.zip', '_2d-label-filt.zip']
FILETYPES_TEST = ['.sens', '.txt', '_vh_clean.ply', '_vh_clean_2.ply']
PREPROCESSED_FRAMES_FILE = ['scannet_frames_25k.zip', '5.6GB']
TEST_FRAMES_FILE = ['scannet_frames_test.zip', '610MB']
LABEL_MAP_FILES = ['scannetv2-labels.combined.tsv', 'scannet-labels.combined.tsv']
RELEASES = ['v2/scans', 'v1/scans']
RELEASES_TASKS = ['v2/tasks', 'v1/tasks']
RELEASES_NAMES = ['v2', 'v1']
RELEASE = RELEASES[0]
RELEASE_TASKS = RELEASES_TASKS[0]
RELEASE_NAME = RELEASES_NAMES[0]
LABEL_MAP_FILE = LABEL_MAP_FILES[0]
RELEASE_SIZE = '1.2TB'
V1_IDX = 1


def get_release_scans(release_file):
    scan_lines = urlopen(release_file)
    scans = []
    for scan_line in scan_lines:
        scan_id = scan_line.decode('utf8').rstrip('\n')
        scans.append(scan_id)
    return scans


def download_release(release_scans, out_dir, file_types, use_v1_sens):
    if len(release_scans) == 0:
        return
    print('Downloading ScanNet ' + RELEASE_NAME + ' release to ' + out_dir + '...')
    for scan_id in release_scans:
        scan_out_dir = os.path.join(out_dir, scan_id)
        download_scan(scan_id, scan_out_dir, file_types, use_v1_sens)
    print('Downloaded ScanNet ' + RELEASE_NAME + ' release.')


def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isfile(out_file):
        print('\t' + url + ' > ' + out_file)
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()
        urllib.request.urlretrieve(url, out_file_tmp)
        #urllib.urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)
    else:
        print('WARNING: skipping download of existing file ' + out_file)

def download_scan(scan_id, out_dir, file_types, use_v1_sens):
    try:
        print('Downloading ScanNet ' + RELEASE_NAME + ' scan ' + scan_id + ' ...')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        for ft in file_types:
            v1_sens = use_v1_sens and ft == '.sens'
            url = BASE_URL + RELEASE + '/' + scan_id + '/' + scan_id + ft if not v1_sens else BASE_URL + RELEASES[V1_IDX] + '/' + scan_id + '/' + scan_id + ft
            out_file = out_dir + '/' + scan_id + ft
            download_file(url, out_file)
        print('Downloaded scan ' + scan_id)
    except:
        FAILED_DOWNLOAD.append(scan_id)


def download_task_data(out_dir):
    print('Downloading ScanNet v1 task data...')
    files = [
        LABEL_MAP_FILES[V1_IDX], 'obj_classification/data.zip',
        'obj_classification/trained_models.zip', 'voxel_labeling/data.zip',
        'voxel_labeling/trained_models.zip'
    ]
    for file in files:
        url = BASE_URL + RELEASES_TASKS[V1_IDX] + '/' + file
        localpath = os.path.join(out_dir, file)
        localdir = os.path.dirname(localpath)
        if not os.path.isdir(localdir):
          os.makedirs(localdir)
        download_file(url, localpath)
    print('Downloaded task data.')


def download_label_map(out_dir):
    print('Downloading ScanNet ' + RELEASE_NAME + ' label mapping file...')
    files = [ LABEL_MAP_FILE ]
    for file in files:
        url = BASE_URL + RELEASE_TASKS + '/' + file
        localpath = os.path.join(out_dir, file)
        localdir = os.path.dirname(localpath)
        if not os.path.isdir(localdir):
          os.makedirs(localdir)
        download_file(url, localpath)
    print('Downloaded ScanNet ' + RELEASE_NAME + ' label mapping file.')

# REFERENCE TO https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py
########################################################################################
#                                                                                      #
#                                      UTILS                                           #
#                                                                                      #
########################################################################################

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

########################################################################################
#                                                                                      #
#                          SCANNET InMemoryDataset DATASET                             #
#                                                                                      #
########################################################################################

class Scannet(InMemoryDataset):

    CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                    'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                    'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
    URLS_METADATA = ["https://raw.githubusercontent.com/facebookresearch/votenet/master/scannet/meta_data/scannetv2-labels.combined.tsv", 
                     "https://raw.githubusercontent.com/facebookresearch/votenet/master/scannet/meta_data/scannetv2_train.txt",
                     "https://raw.githubusercontent.com/facebookresearch/votenet/master/scannet/meta_data/scannetv2_test.txt",
                     "https://raw.githubusercontent.com/facebookresearch/votenet/master/scannet/meta_data/scannetv2_val.txt"]
    VALID_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    
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

    SPLIT = ["train", "val"]

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
        max_num_point=None,
        use_multiprocessing=False,
        process_workers= 4,
        types = [".txt", "_vh_clean_2.ply", "_vh_clean_2.0.010000.segs.json", ".aggregation.json"]
    ):

        if donotcare_class_ids:
            self.DONOTCARE_CLASS_IDS = np.asarray(donotcare_class_ids)
        assert version in ["v2", "v1"], "The version should be either v1 or v2"
        self.version = version
        self.max_num_point = max_num_point
        self.use_instance_labels = use_instance_labels
        self.use_instance_bboxes = use_instance_bboxes
        self.use_multiprocessing = use_multiprocessing
        self.process_workers = process_workers
        self.types = types

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
    def valid_class_ids(self):
        return self.VALID_CLASS_IDS

    @property
    def raw_file_names(self):
        return glob(osp.join(self.raw_dir, "scans", "*"))

    @property
    def processed_file_names(self):
        return ["{}.pt".format(s,) for s in self.SPLIT]

    def download_scans(self):
        release_file = BASE_URL + RELEASE + '.txt'
        release_scans = get_release_scans(release_file)
        file_types = FILETYPES
        release_test_file = BASE_URL + RELEASE + '_test.txt'
        release_test_scans = get_release_scans(release_test_file)
        file_types_test = FILETYPES_TEST
        out_dir_scans = os.path.join(self.raw_dir, 'scans')

        if self.types:  # download file type
            file_types = self.types
            for file_type in file_types:
                if file_type not in FILETYPES:
                    print('ERROR: Invalid file type: ' + file_type)
                    return
            file_types_test = []
            for file_type in file_types:
                if file_type not in FILETYPES_TEST:
                    file_types_test.append(file_type)
        download_label_map(self.raw_dir)
        print('WARNING: You are downloading all ScanNet ' + RELEASE_NAME + ' scans of type ' + file_types[0])
        print('Note that existing scan directories will be skipped. Delete partially downloaded directories to re-download.')
        print('***')
        print('Press any key to continue, or CTRL-C to exit.')
        key = input('')
        if self.version == "v2" and '.sens' in file_types:
            print('Note: ScanNet v2 uses the same .sens files as ScanNet v1: Press \'n\' to exclude downloading .sens files for each scan')
            key = input('')
            if key.strip().lower() == 'n':
                file_types.remove('.sens')
        download_release(release_scans, out_dir_scans, file_types, use_v1_sens=True)
        if self.version == "v2":
            download_label_map(self.raw_dir)

    def download(self):
        self.download_scans()
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
        bbox_mask = np.in1d(instance_bboxes[:,-1], obj_class_ids)
        instance_bboxes = instance_bboxes[bbox_mask,:]

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
        data["x"] = None
        
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

    @staticmethod
    def process_func(id_scan, pre_transform, scannet_dir, scan_name, label_map_file, donotcare_class_ids, max_num_point, obj_class_ids, use_instance_labels=True, use_instance_bboxes=True):
        data = Scannet.read_one_scan(scannet_dir, scan_name, label_map_file, donotcare_class_ids, max_num_point, 
               obj_class_ids, use_instance_labels=use_instance_labels, use_instance_bboxes=use_instance_bboxes)
        if pre_transform:
            data = pre_transform(data)
        log.info("{}| scan_name: {}, data: {}".format(id_scan, scan_name, data))
        return data

    def process(self):
        self.download()
        self.read_from_metadata()
        scannet_dir = osp.join(self.raw_dir, "scans")
        for i, (scan_names, split) in enumerate(zip(self.SCAN_NAMES, self.SPLIT)):
            if not os.path.exists(self.processed_paths[i]):
                scannet_dir = osp.join(self.raw_dir, "scans" if split in ["train", "val"] else "scans_test")
                total = len(scan_names)
                args = [("{}/{}".format(id, total), self.pre_transform, scannet_dir, scan_name, self.LABEL_MAP_FILE, self.DONOTCARE_CLASS_IDS, 
                            self.max_num_point, self.VALID_CLASS_IDS, self.use_instance_labels, self.use_instance_bboxes) for id, scan_name in enumerate(scan_names)]
                if self.use_multiprocessing:
                    with multiprocessing.Pool(processes=self.process_workers) as pool:
                        datas = pool.starmap(Scannet.process_func, args)
                else:
                    datas = []
                    for arg in args:
                        if use_multiprocessing
                        data = Scannet.process_func(*arg)
                        datas.append(data)
                log.info("SAVING TO {}".format(split, self.processed_paths[i]))
                torch.save(self.collate(datas), self.processed_paths[i])

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, len(self))

class ScannetDataset(BaseDataset):

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        use_instance_labels: bool = dataset_opt.use_instance_labels
        use_instance_bboxes: bool = dataset_opt.use_instance_bboxes
        donotcare_class_ids: [] = dataset_opt.donotcare_class_ids
        max_num_point: int = dataset_opt.max_num_point if dataset_opt.max_num_point != "None" else None

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
        valid_class_ids = dataset.train_dataset.valid_class_ids
        return ScannetTracker(dataset, wandb_log=wandb_log, use_tensorboard=tensorboard_log, valid_class_ids=valid_class_ids)

