import os.path as osp
import os
import shutil
import torch

from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader, InMemoryDataset, extract_zip, Data
import torch_geometric.transforms as T
from torch_geometric.io import read_txt_array

from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.metrics.classification_tracker import ClassificationTracker
from torch_points3d.utils.download import download_url

class SampledModelNet(InMemoryDataset):
    r"""The ModelNet10/40 dataset from the `"3D ShapeNets: A Deep
    Representation for Volumetric Shapes"
    <https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf>`_ paper,
    containing sampled CAD models of 40 categories. Each sample contains 10,000
    points uniformly sampled with their normal vector.

    .. note::

        Data objects hold mesh faces instead of edge indices.
        To convert the mesh to a graph, use the
        :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
        To convert the mesh to a point cloud, use the
        :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
        sample a fixed number of points on the mesh faces according to their
        face area.

    Parameters:
    ------------
    root (string): Root directory where the dataset should be saved.
    name (string, optional): The name of the dataset (:obj:`"10"` for
        ModelNet10, :obj:`"40"` for ModelNet40). (default: :obj:`"10"`)
    train (bool, optional): If :obj:`True`, loads the training dataset,
        otherwise the test dataset. (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    """

    url = "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip"

    def __init__(self, root, name="10", train=True, transform=None, pre_transform=None, pre_filter=None):
        assert name in ["10", "40"]
        self.name = name
        super(SampledModelNet, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]

    @property
    def processed_file_names(self):
        return ["training_{}.pt".format(self.name), "test_{}.pt".format(self.name)]

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        folder = osp.join(self.root, "modelnet40_normal_resampled")
        shutil.rmtree(self.raw_dir)
        os.rename(folder, self.raw_dir)

    def process(self):
        torch.save(self.process_set("train"), self.processed_paths[0])
        torch.save(self.process_set("test"), self.processed_paths[1])

    def process_set(self, dataset):
        with open(osp.join(self.raw_dir, "modelnet{}_shape_names.txt".format(self.name)), "r") as f:
            categories = f.read().splitlines()
            categories = sorted(categories)
        with open(osp.join(self.raw_dir, "modelnet{}_{}.txt".format(self.name, dataset)), "r") as f:
            split_objects = f.read().splitlines()

        data_list = []
        for target, category in enumerate(categories):
            folder = osp.join(self.raw_dir, category)
            category_ojects = filter(lambda o: category in o, split_objects)
            paths = ["{}/{}.txt".format(folder, o.strip()) for o in category_ojects]
            for path in paths:
                raw = read_txt_array(path, sep=",")
                data = Data(pos=raw[:, :3], norm=raw[:, 3:], y=torch.tensor([target]))
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)

    def __repr__(self):
        return "{}{}({})".format(self.__class__.__name__, self.name, len(self))


class ModelNetDataset(BaseDataset):

    AVAILABLE_NUMBERS = ["10", "40"]

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        number = dataset_opt.number
        if str(number) not in self.AVAILABLE_NUMBERS:
            raise Exception("Only ModelNet10 and ModelNet40 are available")
        self.train_dataset = SampledModelNet(
            self._data_path,
            name=str(number),
            train=True,
            transform=self.train_transform,
            pre_transform=self.pre_transform,
        )
        self.test_dataset = SampledModelNet(
            self._data_path,
            name=str(number),
            train=False,
            transform=self.test_transform,
            pre_transform=self.pre_transform,
        )

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker
        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return ClassificationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
