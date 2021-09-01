import torch
from torch._C import dtype
from torch_geometric.data import InMemoryDataset, Data
from plyfile import PlyData
import numpy as np
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from concurrent.futures import ProcessPoolExecutor as Executor

def process_file(file):
    print("Start", file)
    plydata = PlyData.read(file)
    print("Read data", file)
    points = plydata['vertex']
    pos = np.stack((points['x'], points['y'], points['z']), axis=1)
    normal = np.stack((points['nx'], points['ny'], points['nz']), axis=1)
    rgb = np.stack((points['red'], points['green'], points['blue']), axis=1)
    y = points['label'] - 1
    print("Get data", file)
    
    data = Data(pos=torch.as_tensor(torch.from_numpy(pos.copy()), dtype=torch.float), normal=torch.as_tensor(torch.from_numpy(normal.copy()), dtype=torch.float), rgb=torch.from_numpy(rgb.copy()), y=torch.as_tensor(torch.from_numpy(y.copy()), dtype=torch.long))
    print("End", file)
    return data

class SUMPointCloudDataset(InMemoryDataset):
    def __init__(self, root, test=False, validate=False, transform=None, pre_transform=None):
        self.test = test
        self.validate = validate
        super(SUMPointCloudDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.test == True:
            return [
                "test/Tile_+1984_+2688_pc.ply",
                "test/Tile_+1984_+2693_pc.ply",
                "test/Tile_+1985_+2690_pc.ply",
                "test/Tile_+1986_+2689_pc.ply",
                "test/Tile_+1986_+2691_pc.ply",
                "test/Tile_+1987_+2694_pc.ply",
                "test/Tile_+1988_+2689_pc.ply",
                "test/Tile_+1989_+2692_pc.ply",
                "test/Tile_+1990_+2688_pc.ply",
                "test/Tile_+1990_+2691_pc.ply",
                "test/Tile_+1990_+2695_pc.ply",
                "test/Tile_+1991_+2693_pc.ply"
            ]
        elif self.validate == True:
            return [
                "validate/Tile_+1984_+2689_pc.ply",
                "validate/Tile_+1984_+2691_pc.ply",
                "validate/Tile_+1984_+2694_pc.ply",
                "validate/Tile_+1985_+2688_pc.ply",
                "validate/Tile_+1986_+2692_pc.ply",
                "validate/Tile_+1986_+2695_pc.ply",
                "validate/Tile_+1987_+2690_pc.ply",
                "validate/Tile_+1988_+2693_pc.ply",
                "validate/Tile_+1989_+2690_pc.ply",
                "validate/Tile_+1989_+2694_pc.ply",
                "validate/Tile_+1990_+2692_pc.ply",
                "validate/Tile_+1991_+2689_pc.ply"
            ]
        else:
            return [
                "train/Tile_+1984_+2690_pc.ply",
                "train/Tile_+1984_+2692_pc.ply",
                "train/Tile_+1984_+2695_pc.ply",
                "train/Tile_+1985_+2689_pc.ply",
                "train/Tile_+1985_+2691_pc.ply",
                "train/Tile_+1985_+2692_pc.ply",
                "train/Tile_+1985_+2693_pc.ply",
                "train/Tile_+1985_+2694_pc.ply",
                "train/Tile_+1985_+2695_pc.ply",
                "train/Tile_+1986_+2688_pc.ply",
                "train/Tile_+1986_+2690_pc.ply",
                "train/Tile_+1986_+2693_pc.ply",
                "train/Tile_+1986_+2694_pc.ply",
                "train/Tile_+1987_+2688_pc.ply",
                "train/Tile_+1987_+2689_pc.ply",
                "train/Tile_+1987_+2691_pc.ply",
                "train/Tile_+1987_+2692_pc.ply",
                "train/Tile_+1987_+2693_pc.ply",
                "train/Tile_+1987_+2695_pc.ply",
                "train/Tile_+1988_+2688_pc.ply",
                "train/Tile_+1988_+2690_pc.ply",
                "train/Tile_+1988_+2691_pc.ply",
                "train/Tile_+1988_+2692_pc.ply",
                "train/Tile_+1988_+2694_pc.ply",
                "train/Tile_+1988_+2695_pc.ply",
                "train/Tile_+1989_+2688_pc.ply",
                "train/Tile_+1989_+2689_pc.ply",
                "train/Tile_+1989_+2691_pc.ply",
                "train/Tile_+1989_+2693_pc.ply",
                "train/Tile_+1989_+2695_pc.ply",
                "train/Tile_+1990_+2689_pc.ply",
                "train/Tile_+1990_+2690_pc.ply",
                "train/Tile_+1990_+2693_pc.ply",
                "train/Tile_+1990_+2694_pc.ply",
                "train/Tile_+1991_+2688_pc.ply",
                "train/Tile_+1991_+2690_pc.ply",
                "train/Tile_+1991_+2691_pc.ply",
                "train/Tile_+1991_+2692_pc.ply",
                "train/Tile_+1991_+2694_pc.ply",
                "train/Tile_+1991_+2695_pc.ply"
            ]

    @property
    def processed_file_names(self):
        if self.test == True:
            return ["test_data.pt"]
        elif self.validate == True:
            return ["validate_data.pt"]
        else:
            return ["train_data.pt"]

    def download(self):
        print("Can't find ", self.raw_paths[0], ", use Mesh-2-Point-Cloud to create point-cloud files")
        exit(1)

    def process(self):
        # Read data into huge `Data` list.

        with Executor(max_workers=40) as executor:
            executors = [executor.submit(process_file, file) for file in self.raw_paths]
        data_list = [executor.result() for executor in executors]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            print("Pretransforming data")
            with Executor(max_workers=40) as executor:
                executors = [executor.submit(self.pre_transform, data) for data in data_list]
            data_list = [executor.result() for executor in executors]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class SUMDataset(BaseDataset):
    INV_OBJECT_LABEL = {
        -1: "unclassified",
        0: "ground",
        1: "vegetation",
        2: "building",
        3: "water",
        4: "car",
        5: "boat"
    }

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        self.train_dataset = SUMPointCloudDataset(
            self._data_path,
            pre_transform=self.pre_transform,
            transform=self.train_transform,
        )
        self.test_dataset = SUMPointCloudDataset(
            self._data_path,
            test=True,
            pre_transform=self.pre_transform,
            transform=self.test_transform,
        )
        self.val_dataset = SUMPointCloudDataset(
            self._data_path,
            validate=True,
            pre_transform=self.pre_transform,
            transform=self.val_transform,
        )


    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log, ignore_label=-1)