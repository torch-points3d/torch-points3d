import torch
from torch_geometric.data import Dataset, Data, Batch, dataset
from plyfile import PlyData
import numpy as np
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.core.data_transform import Select
from concurrent.futures import ProcessPoolExecutor as Executor
from random import randint
from sklearn.neighbors import KDTree

class SUMPointCloudDataset(Dataset):
    r"""SUM Dataset in point cloud format

    Args:
        root (string): Root directory where the dataset should be saved.
        length (int, optional): Number of point cloud to be sample.
            (default: :int:1000)
        cloud_size (int, optional): Number of point in out sample cloud. (default: :int:4096)
        test (bool): Load test dataset. (default: :bool:False)
        validate (bool): Load validate dataset. (default: :bool:False)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        process_workers (int, optional): Number of process to use for pre_transform and
            transform. (default: :obj:`None`)
        leaf_size (int, optional): Number of point in KDTree leaf. (default: :int:50)
    """

    def __init__(self, root, length=1000, cloud_size=4096, test=False, validate=False, transform=None, pre_transform=None, process_workers=1, leaf_size=50):
        self.test = test
        self.validate = validate
        self.process_workers = process_workers
        self.length = length
        self.leaf_size = leaf_size
        self.cloud_size = cloud_size
        super(SUMPointCloudDataset, self).__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

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

    @staticmethod
    def process_one(file, leaf_size):
        print("Starting", file)
        plydata = PlyData.read(file)
        print("Reading data", file)
        points = plydata['vertex']
        pos = np.stack((points['x'], points['y'], points['z']), axis=1)
        normal = np.stack((points['nx'], points['ny'], points['nz']), axis=1)
        rgb = np.stack((points['red'], points['green'], points['blue']), axis=1)
        y = points['label'] - 1
        print("Get data", file)
        
        data = Data(pos=torch.as_tensor(torch.from_numpy(pos.copy()), dtype=torch.float), normal=torch.as_tensor(torch.from_numpy(normal.copy()), dtype=torch.float), rgb=torch.from_numpy(rgb.copy()), y=torch.as_tensor(torch.from_numpy(y.copy()), dtype=torch.long))
        print("Ending", file)
        return data

    def process(self):
        # Read data into huge `Data` list.
        if self.process_workers > 1:
            with Executor(max_workers=self.process_workers) as executor:
                results = [executor.submit(self.process_one, file, self.leaf_size) for file in self.raw_paths]
            data_list = [result.result() for result in results]
        else:
            data_list = [self.process_one(file) for file in self.raw_paths]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            print("Pretransforming data")
            if self.process_workers > 1:
                with Executor(max_workers=self.process_workers) as executor:
                    results = [executor.submit(self.pre_transform, data) for data in data_list]
                data_list = [result.result() for result in results]
            else:
                data_list = [self.pre_transform(data) for data in data_list]
            if isinstance(data_list[0], list):
                data_list=[data for datas in data_list for data in datas]

        data = Batch.from_data_list(data_list)
        torch.save(data, self.processed_paths[0])

    def len(self):
        return self.length

    def get(self, idx):
        cloud_idx = randint(0, len(self.data)-1)
        data = self.data.get_example(cloud_idx)
        if (data.num_nodes <= self.cloud_size):
            return data
        else:
            node_idx = randint(0,data.num_nodes-1)
            if not hasattr(data, "kd_tree"):
                data.kd_tree = KDTree(np.asarray(data.pos), leaf_size=self.leaf_size)
            index = data.kd_tree.query(data.pos[node_idx].reshape(1,3), k=self.cloud_size, return_distance=False).flatten()
            transform = Select(index)
            data = transform(data)
        return data

    @property
    def num_classes(self) -> int:
        return 7


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

        process_workers: int = dataset_opt.process_workers if hasattr(dataset_opt,'process_workers') else 0

        self.train_dataset = SUMPointCloudDataset(
            self._data_path,
            length = dataset_opt.length,
            cloud_size = dataset_opt.cloud_size,
            pre_transform=self.pre_transform,
            transform=self.train_transform,
            process_workers=process_workers,
            leaf_size = dataset_opt.leaf_size
        )
        self.test_dataset = SUMPointCloudDataset(
            self._data_path,
            length = dataset_opt.length,
            cloud_size = dataset_opt.cloud_size,
            test=True,
            pre_transform=self.pre_transform,
            transform=self.test_transform,
            process_workers=process_workers,
            leaf_size = dataset_opt.leaf_size
        )
        self.val_dataset = SUMPointCloudDataset(
            self._data_path,
            length = dataset_opt.length,
            cloud_size = dataset_opt.cloud_size,
            validate=True,
            pre_transform=self.pre_transform,
            transform=self.val_transform,
            process_workers=process_workers,
            leaf_size = dataset_opt.leaf_size
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