"""
Dataset related entries
"""
labels = { 
  0 : "unlabeled",
  1 : "outlier",
  10: "car",
  11: "bicycle",
  13: "bus",
  15: "motorcycle",
  16: "on-rails",
  18: "truck",
  20: "other-vehicle",
  30: "person",
  31: "bicyclist",
  32: "motorcyclist",
  40: "road",
  44: "parking",
  48: "sidewalk",
  49: "other-ground",
  50: "building",
  51: "fence",
  52: "other-structure",
  60: "lane-marking",
  70: "vegetation",
  71: "trunk",
  72: "terrain",
  80: "pole",
  81: "traffic-sign",
  99: "other-object",
  252: "moving-car",
  253: "moving-bicyclist",
  254: "moving-person",
  255: "moving-motorcyclist",
  256: "moving-on-rails",
  257: "moving-bus",
  258: "moving-truck",
  259: "moving-other-vehicle"
}

color_map = {
  0 : [0, 0, 0],
  1 : [0, 0, 255],
  10: [245, 150, 100],
  11: [245, 230, 100],
  13: [250, 80, 100],
  15: [150, 60, 30],
  16: [255, 0, 0],
  18: [180, 30, 80],
  20: [255, 0, 0],
  30: [30, 30, 255],
  31: [200, 40, 255],
  32: [90, 30, 150],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [75, 0, 175],
  50: [0, 200, 255],
  51: [50, 120, 255],
  52: [0, 150, 255],
  60: [170, 255, 150],
  70: [0, 175, 0],
  71: [0, 60, 135],
  72: [80, 240, 150],
  80: [150, 240, 255],
  81: [0, 0, 255],
  99: [255, 255, 50],
  252: [245, 150, 100],
  256: [255, 0, 0],
  253: [200, 40, 255],
  254: [30, 30, 255],
  255: [90, 30, 150],
  257: [250, 80, 100],
  258: [180, 30, 80],
  259: [255, 0, 0]
}
# ratio of points in a class with respect to the total number
content = {
  0: 0.018889854628292943,
  1: 0.0002937197336781505,
  10: 0.040818519255974316,
  11: 0.00016609538710764618,
  13: 2.7879693665067774e-05,
  15: 0.00039838616015114444,
  16: 0.0,
  18: 0.0020633612104619787,
  20: 0.0016218197275284021,
  30: 0.00017698551338515307,
  31: 1.1065903904919655e-08,
  32: 5.532951952459828e-09,
  40: 0.1987493871255525,
  44: 0.014717169549888214,
  48: 0.14392298360372,
  49: 0.0039048553037472045,
  50: 0.1326861944777486,
  51: 0.0723592229456223,
  52: 0.002395131480328884,
  60: 4.7084144280367186e-05,
  70: 0.26681502148037506,
  71: 0.006035012012626033,
  72: 0.07814222006271769,
  80: 0.002855498193863172,
  81: 0.0006155958086189918,
  99: 0.009923127583046915,
  252: 0.001789309418528068,
  253: 0.00012709999297008662,
  254: 0.00016059776092534436,
  255: 3.745553104802113e-05,
  256: 0.0,
  257: 0.00011351574470342043,
  258: 0.00010157861367183268,
  259: 4.3840131989471124e-05
}

"""
some objects are not identifiable from a single scan and hence are mapped to their closest class
"""
learning_map = {
  0 : 0,     # "unlabeled"
  1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
  10: 1,     # "car"
  11: 2,     # "bicycle"
  13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
  15: 3,     # "motorcycle"
  16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
  18: 4,     # "truck"
  20: 5,     # "other-vehicle"
  30: 6,     # "person"
  31: 7,     # "bicyclist"
  32: 8,     # "motorcyclist"
  40: 9,     # "road"
  44: 10,    # "parking"
  48: 11,    # "sidewalk"
  49: 12,    # "other-ground"
  50: 13,    # "building"
  51: 14,    # "fence"
  52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
  60: 9,     # "lane-marking" to "road" ---------------------------------mapped
  70: 15,    # "vegetation"
  71: 16,    # "trunk"
  72: 17,    # "terrain"
  80: 18,    # "pole"
  81: 19,    # "traffic-sign"
  99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
  252: 1,    # "moving-car" to "car" ------------------------------------mapped
  253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
  254: 6,    # "moving-person" to "person" ------------------------------mapped
  255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
  256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
  257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
  258: 4,    # "moving-truck" to "truck" --------------------------------mapped
  259: 5     # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}
# invert above feature map
learning_map_inv = {
  0: 0,      # "unlabeled", and others ignored
  1: 10,     # "car"
  2: 11,     # "bicycle"
  3: 15,     # "motorcycle"
  4: 18,     # "truck"
  5: 20,     # "other-vehicle"
  6: 30,     # "person"
  7: 31,     # "bicyclist"
  8: 32,     # "motorcyclist"
  9: 40,     # "road"
  10: 44,    # "parking"
  11: 48,    # "sidewalk"
  12: 49,    # "other-ground"
  13: 50,    # "building"
  14: 51,    # "fence"
  15: 70,    # "vegetation"
  16: 71,    # "trunk"
  17: 72,    # "terrain"
  18: 80,    # "pole"
  19: 81     # "traffic-sign"
}

# Ignore classes
learning_ignore = { 
  0: True,      # "unlabeled", and others ignored
  1: False,     # "car"
  2: False,     # "bicycle"
  3: False,     # "motorcycle"
  4: False,     # "truck"
  5: False,     # "other-vehicle"
  6: False,     # "person"
  7: False,     # "bicyclist"
  8: False,     # "motorcyclist"
  9: False,     # "road"
  10: False,    # "parking"
  11: False,    # "sidewalk"
  12: False,    # "other-ground"
  13: False,    # "building"
  14: False,    # "fence"
  15: False,    # "vegetation"
  16: False,    # "trunk"
  17: False,    # "terrain"
  18: False,    # "pole"
  19: False     # "traffic-sign"
}
# sequence numbers
split = { 
  "train" : [0,1,2,3,4,5,6,7,9,10],
  "valid" : [8],
  "test"  : [11,12,13,14,15,16,17,18,19,20,21]
}

sensor_config = {
    "name": "HDL64",
    "type": "spherical",
    "fov_up": 3,
    "fov_down": -25
}
img_prop = {
    "width": 1024,
    "height": 64,
    # range, x, y, z signal
    "img_means": [12.12, 10.88, 0.23, -1.04, 0.21],
    "img_stds": [12.32, 11,47, 6,91, 0.86, 0.16] 
}

import numpy as np
from torch.utils.data.dataset import Dataset
import os
from glob import glob
import torch

class LidarScan:
    """
    class for handling lidar scans and corresponding labels
    inputs: H, W - height, width of projected image
            FOV_UP, FOV_DOWN - field of view in upward and downward direction for a vertical lidar
    """
    def __init__(self, H, W, FOV_UP, FOV_DOWN):
    
        self.H = H
        self.W = W
        self.FOV_UP = FOV_UP/180*np.pi # convert to radians
        self.FOV_DOWN = FOV_DOWN/180*np.pi
        self.FOV = abs(self.FOV_UP) + abs(self.FOV_DOWN)
    
    def populate_colormap(self):
        """
        populate colormaps - both reduced and complete color map in RGB.
        """
        self.colormap = np.zeros((max(list(learning_map.keys()))+1, 3), dtype=np.float32)
        for key, value in color_map.items():
            value = [value[i] for i in [2,1,0]]
            self.colormap[key] = np.array(value, np.float32) / 255.0
        self.reduced_colormap = np.zeros((max(list(learning_map_inv.keys()))+1, 3), dtype=np.float32)
        for red_cl, map_cl in learning_map_inv.items():
            self.reduced_colormap[red_cl] = self.colormap[map_cl]
    
    def open_scan(self, file_name):
        """
        open scan file - [x, y, z, remissions]
        returns point coordinates, remissions
        """
        scan = np.fromfile(file_name, dtype=np.float32)
        scan = scan.reshape((-1, 4)) # just for the sake of it
        points = scan[:, 0:3]
        remissions = scan[:, 3]
        return points, remissions
    
    def open_label(self, file_name):
        """
        open label file - [semantic label(first 16 bits), instance label(last 16 bits)]
        returns semantic label, instance label
        """
        label = np.fromfile(file_name, dtype=np.uint32)
        label = label.reshape((-1)) # again for the sake of it
        semantic_label = label & 0xFFFF
        instance_label = label >> 16
        return semantic_label, instance_label
    
    def project_scan(self, unproj_points, unproj_remissions):
        """
        project the lidar scan to a spherical image
        return ordered points, ordered remissions, projected range, projected xyz, projected remissions, projected index, projected mask, proj_x (index mapping to image width), proj_y (index mapping to image height)
        """
        depth = np.linalg.norm(unproj_points, 2, axis=1) # calculate depth
        # get x, y, z seperately
        scan_x, scan_y, scan_z = unproj_points[:, 0], unproj_points[:, 1], unproj_points[:, 2]
        yaw = -np.arctan2(scan_y, scan_x) # yaw angle -> -tan-1(left/forward) -> (-pi,pi)
        pitch = np.arcsin(scan_z/depth) # pitch angle -> sin-1(top/depth)

        proj_x = 0.5*(yaw/np.pi+1.0) # yaw/pi + 1 --> (0,2)/2 -> (0,1)
        proj_x = np.floor(proj_x*self.W) # convert to (0,W) and find nearest integer
        proj_x = np.minimum(self.W-1, proj_x) # ensure values lie in range
        proj_x = np.maximum(0, proj_x).astype(np.int32) 

        proj_y = (abs(self.FOV_UP)-pitch)/self.FOV # convert pitch values to 0,1
        proj_y = np.floor(proj_y*self.H) # convert to (0,H) and find nearest integer
        proj_y = np.minimum(self.H-1, proj_y) # ensure values lie in range
        proj_y = np.maximum(0, proj_y).astype(np.int32) 

        # order by decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = unproj_points[order]
        remissions = unproj_remissions[order]
        proj_x = proj_x[order]
        proj_y = proj_y[order]

        # initializing empty arrays
        proj_range = np.full((self.H, self.W), -1, dtype=np.float32)
        proj_xyz = np.full((self.H, self.W, 3), -1, dtype=np.float32)
        proj_remissions = np.full((self.H, self.W), -1, dtype=np.float32)
        proj_index = np.full((self.H, self.W), -1, dtype=np.int32)

        # assigning values!
        proj_range[proj_y, proj_x] = depth
        proj_xyz[proj_y, proj_x] = points
        proj_remissions[proj_y, proj_x] = remissions
        proj_index[proj_y, proj_x] = indices
        proj_mask = (proj_index >= 0).astype(np.bool)

        return points, remissions, proj_range, proj_xyz, proj_remissions, proj_index, proj_mask, proj_x, proj_y

    def project_sem_label(self, proj_index, proj_mask, unproj_sem_label):
        """
        project label for corresponding scan
        """
        proj_sem_label = np.zeros((self.H, self.W), dtype=np.int32)
        proj_sem_label[proj_mask] = unproj_sem_label[proj_index[proj_mask]]
        return proj_sem_label
    
    def colorise_points(self, unproj_label, reduced=True):
        """
        returns color labels for unprojected points. reduced=True/False based on reduced color map requirement
        """
        if reduced:
            unproj_label = np.vectorize(learning_map.get)(unproj_label)
            colored_unproj_label = self.reduced_colormap[unproj_label]
        else:
            colored_unproj_label = self.colormap[unproj_label]
        return colored_unproj_label
    
    def unproject_labels(self, proj_label, proj_x, proj_y):
        """
        unproject projected labels
        """
        return proj_label[proj_y, proj_x]
    
    def colorise_projection(self, proj_label, proj_mask, reduced=True):
        """
        colorise projected image
        """
        proj_color_labels = np.zeros((self.H, self.W, 3), dtype=np.float)
        if reduced:
            proj_label = np.vectorize(learning_map.get)(proj_label)
            proj_color_labels[proj_mask] = self.reduced_colormap[proj_label[proj_mask]]
        else:
            proj_color_labels[proj_mask] = self.colormap[proj_label[proj_mask]]
        return proj_color_labels


class SemanticKitti(Dataset):
    def __init__(self, base_dir, sensor_config, mode=None):
        self.base_dir = base_dir
        if mode == None:
            self.mode = "train"
        else:
            self.mode = mode
        self.train_scans, self.train_labels = self.get_filenames(split["train"])
        self.val_scans, self.val_labels = self.get_filenames(split["valid"])
        self.test_scans = self.get_filenames(split["test"], False)
        self.lidar_scan = LidarScan(img_prop["height"], img_prop["width"], sensor_config["fov_up"], sensor_config["fov_down"])
        self.sensor_means = torch.tensor(sensor_config["img_means"], dtype=torch.float)
        self.sensor_stds = torch.tensor(sensor_config["img_stds"], dtype=torch.float)

    def get_filenames(self, sequences, labels=True):
        velodyne_files = []
        label_files = []
        for sequence in sequences:
            velodyne_dir = os.path.join(self.base_dir, "{0:02d}".format(int(sequence)), "velodyne")
            v = sorted(glob(os.path.join(velodyne_dir, '*.bin')))
            velodyne_files.extend(v)
            if labels:
                label_dir = os.path.join(self.base_dir, "{0:02d}".format(int(sequence)), "labels")
                l = sorted(glob(os.path.join(label_dir, '*.label')))
                label_files.extend(l)
                if len(v) != len(l):
                    raise Exception("Sequence {} has unequal label and velodyne files")
            print("Sequence {} has {} files".format(sequence, len(v)))
        if labels:
            return velodyne_files, label_files
        else:
            return velodyne_files

    def __getitem__(self, index):
        if self.mode == "train":
            velodyne_file = self.train_scans[index]
            label_file = self.train_labels[index]
        elif self.mode == "val":
            velodyne_file = self.val_scans[index]
            label_file = self.val_labels[index]
        else:
            velodyne_file = self.test_scans[index]

        points, remissions = self.lidar_scan.open_scan(velodyne_file)
        _, _, proj_range, proj_xyz, proj_remissions, _, proj_mask, _, _ = self.lidar_scan.project_scan(points, remissions)
        proj_range, proj_xyz, proj_remissions = torch.from_numpy(proj_range).clone(), torch.from_numpy(proj_xyz).clone(), torch.from_numpy(proj_remissions).clone()
        proj_inp = torch.cat([proj_range.unsqueeze(0).clone(), proj_xyz.clone().permute(2, 0, 1), proj_remissions.unsqueeze(0).clone()])
        proj_inp = (proj_inp - self.sensor_means[:, None, None])/self.sensor_stds[:, None, None]

        if self.mode == "train" or self.mode == "val":
            sem_label, _ = self.lidar_scan.open_label(label_file)
            sem_label = np.vectorize(learning_map.get)(sem_label)
            proj_sem_label = self.lidar_scan.project_sem_label(sem_label)
            proj_sem_label = torch.from_numpy(proj_sem_label).clone()
            return proj_inp, proj_sem_label
        else:
            return proj_inp
   
    def __len__(self):
        if self.mode == "train":
            return len(self.train_scans)
        elif self.mode == "val":
            return len(self.val_scans)
        else:
            return len(self.test_scans)









 
    





    
