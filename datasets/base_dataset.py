from abc import ABC, abstractmethod
import logging
from datasets.transforms import MultiScaleTransform
import torch
import torch_geometric
from torch_geometric.data import Data

# A logger for this file
log = logging.getLogger(__name__)

class Batch(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(Batch, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list, follow_batch=[], batch_transform=None):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`."""

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = Batch()

        for key in keys:
            batch[key] = []

        for key in follow_batch:
            batch['{}_batch'.format(key)] = []

        cumsum = {}
        batch.batch = []
        for i, data in enumerate(data_list):
            for key in data.keys:
                item = data[key] + cumsum.get(key, 0)
                if key in cumsum:
                    cumsum[key] += data.__inc__(key, item)
                else:
                    cumsum[key] = data.__inc__(key, item)
                batch[key].append(item)

            for key in follow_batch:
                size = data[key].size(data.__cat_dim__(key, data[key]))
                item = torch.full((size, ), i, dtype=torch.long)
                batch['{}_batch'.format(key)].append(item)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes, ), i, dtype=torch.long)
                batch.batch.append(item)

        if num_nodes is None:
            batch.batch = None

        for key in batch.keys:
            item = batch[key][0]
            if torch.is_tensor(item):
                batch[key] = torch.cat(
                    batch[key], dim=data_list[0].__cat_dim__(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])
            else:
                raise ValueError('Unsupported attribute type.')

        if torch_geometric.is_debug_enabled():
            batch.debug()

        if batch_transform is None:
            return batch.contiguous()
        else:
            return batch_transform(batch).contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """

    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 follow_batch=[],
                 batch_transform=None,
                 **kwargs):
        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn = lambda data_list: Batch.from_data_list(
                data_list, follow_batch, batch_transform),
            **kwargs)

class BaseDataset():
    def __init__(self, dataset_opt, training_opt):
        self.dataset_opt = dataset_opt
        self.training_opt = training_opt
        self.strategies = {}

    def create_dataloaders(self, train_dataset,  test_dataset, validation=None):
        self._num_classes = train_dataset.num_classes
        self._train_loader = DataLoader(train_dataset, batch_size=self.training_opt.batch_size, shuffle=self.training_opt.shuffle,
                                num_workers=self.training_opt.num_workers)

        self._test_loader = DataLoader(test_dataset, batch_size=self.training_opt.batch_size, shuffle=False,
                                num_workers=self.training_opt.num_workers)
    @abstractmethod
    def test_dataloader(self):
        pass

    @abstractmethod
    def train_dataloader(self):
        pass

    @property
    @abstractmethod
    def num_classes(self):
        pass

    def _set_multiscale_transform(self, batch_transform):
        for attr_name, attr in self.__dict__.items():
            if "loader" in attr_name and isinstance(attr, DataLoader):
                collate_fn=lambda data_list: Batch.from_data_list(
                data_list, [], batch_transform)
                setattr(attr, "collate_fn", collate_fn)

    def set_strategies(self, strategies, precompute_multi_scale=False):
        batch_transform = MultiScaleTransform(strategies, precompute_multi_scale)
        self._set_multiscale_transform(batch_transform)


    
