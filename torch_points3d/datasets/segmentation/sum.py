import torch
from torch_geometric.data import Dataset, Data, Batch, dataset
from plyfile import PlyData
import numpy as np
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.metrics.sum_tracker import SUMTracker
from torch_points3d.core.data_transform import Select, GridNeigborsSampling, PointCloudFusion
from concurrent.futures import ProcessPoolExecutor as Executor
from random import randint
from sklearn.neighbors import KDTree
import pickle

class SUMPointCloudDataset(Dataset):
    r"""SUM Dataset in point cloud format
    If length is not None, length point clouds of size cloud_size will be randomly sample.
    Otherwise, sample_spacing must be provided, and point clouds of size cloud_size will be sampled regularly,
    with a center-to-center spacing of sample_spacing.

    Args:
        root (string): Root directory where the dataset should be saved.
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
        length (int, optional): Number of point cloud to be sample.
            (default: :obj:`None`)
        sample_spacing (float, optional): Distance between two sample centers.
            (default: :obj:`None`)
    """

    def __init__(self, root:str, cloud_size:int=4096, test:bool=False, validate:bool=False, transform=None, pre_transform=None, leaf_size:int=50, process_workers:int=1, length:int=None, sample_spacing:float=None):
        if length is None and sample_spacing is None:
            raise NotImplementedError("length or sample_spacing need to be specified")

        self.length = length
        self.sample_spacing = sample_spacing
        self.cloud_size = cloud_size
        self.test = test
        self.validate = validate
        self.leaf_size = leaf_size
        self.process_workers = process_workers
        super(SUMPointCloudDataset, self).__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])
        self.kd_tree = pickle.load(open(self.processed_paths[1], "rb"))
        if length is None:
            self.samples = torch.load(self.processed_paths[2])


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
            files = ["test_data.pt", "test_kdtree.pkl"]
            if self.length is None:
                files.append("test_samples.pt")
        elif self.validate == True:
            files = ["validate_data.pt", "validate_kdtree.pkl"]
            if self.length is None:
                files.append("validate_samples.pt")
        else:
            files = ["train_data.pt", "train_kdtree.pkl"]
            if self.length is None:
                files.append("train_samples.pt")
        return files

    def download(self):
        print("Can't find ", self.raw_paths[0], ", use Mesh-2-Point-Cloud to create point-cloud files")
        exit(1)

    @staticmethod
    def process_one(file:str):
        plydata = PlyData.read(file)
        points = plydata['vertex']
        pos = np.stack((points['x'], points['y'], points['z']), axis=1)
        normal = np.stack((points['nx'], points['ny'], points['nz']), axis=1)
        rgb = np.stack((points['red'], points['green'], points['blue']), axis=1)
        y = points['label'] - 1
        fid = points['face_id']
        fnormal = np.stack((points['fnx'], points['fny'], points['fnz']), axis=1)
        
        data = Data(
            pos=torch.from_numpy(pos.astype(np.float32)),
            normal=torch.from_numpy(normal.astype(np.float32)),
            rgb=torch.from_numpy(rgb.copy()),
            y=torch.from_numpy(y.astype(np.long)),
            fid=torch.from_numpy(fid.astype(np.long)),
            fnormal=torch.from_numpy(fnormal.astype(np.float32)))
        return data

    def process(self):
        # Read data into huge `Data` list.
        print("Reading PLY files")
        if self.process_workers > 1:
            with Executor(max_workers=self.process_workers) as executor:
                results = [executor.submit(self.process_one, file) for file in self.raw_paths]
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

        print("Merging data")
        data = PointCloudFusion()(data_list)

        print("Computing KDTree")
        kd_tree = KDTree(data.pos.numpy(), leaf_size=self.leaf_size)
        pickle.dump( kd_tree, open(self.processed_paths[1], "wb"))

        if self.length is not None:
            print("Indexing class for random sampling")
            classes, indexes = torch.unique(data.y, return_inverse=True)
            class_index = []
            for i in range(len(classes)):
                if classes[i] >= 0:
                    class_index.append((indexes == i).nonzero().flatten())
            data.class_index = torch.cat(class_index)
            data.class_size = torch.tensor([0]+[torch.numel(t) for t in class_index]).cumsum(0)
        else:
            print("Computing samples")
            data.kd_tree = kd_tree
            samples = GridNeigborsSampling(self.cloud_size, self.sample_spacing, delattr_kd_tree=True, center=False)(data)
            torch.save(samples, self.processed_paths[2])

        torch.save(data, self.processed_paths[0])


    def len(self) -> int:
        if self.length is not None:
            return self.length
        else:
            return len(self.samples)

    def get(self, idx):
        if self.length is not None:
            if (self.data.num_nodes <= self.cloud_size or self.cloud_size < 1):
                data = self.data.clone()
            else:
                class_idx = randint(0, len(self.data.class_size)-2) # choose a class
                raw_idx = randint(self.data.class_size[class_idx], self.data.class_size[class_idx+1]-1) # choose a point in the class
                node_idx = self.data.class_index[raw_idx]
                index = self.kd_tree.query(self.data.pos[node_idx].reshape(1,3), k=self.cloud_size, return_distance=False).flatten() # Comput k-neighbors
                data = Select(index)(self.data) # Extract cloud point
            delattr(data, "class_size") # Remove class size
            delattr(data, "class_index") # Remove class index
            return data
        else:
            return self.samples[idx]

    @property
    def num_classes(self) -> int:
        return 6


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

        process_workers: int = dataset_opt.process_workers if hasattr(dataset_opt,'process_workers') else 1

        self.train_dataset = SUMPointCloudDataset(
            self._data_path,
            cloud_size = dataset_opt.cloud_size,
            pre_transform=self.pre_transform,
            transform=self.train_transform,
            leaf_size=dataset_opt.leaf_size,
            process_workers=process_workers,
            length = dataset_opt.length
        )
        self.test_dataset = SUMPointCloudDataset(
            self._data_path,
            cloud_size = dataset_opt.cloud_size,
            test=True,
            pre_transform=self.pre_transform,
            transform=self.test_transform,
            leaf_size=dataset_opt.leaf_size,
            process_workers=process_workers,
            sample_spacing=dataset_opt.sample_spacing
        )
        self.val_dataset = SUMPointCloudDataset(
            self._data_path,
            cloud_size = dataset_opt.cloud_size,
            validate=True,
            pre_transform=self.pre_transform,
            transform=self.val_transform,
            leaf_size=dataset_opt.leaf_size,
            process_workers=process_workers,
            length = dataset_opt.length
        )


    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return SUMTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log, ignore_label=-1)