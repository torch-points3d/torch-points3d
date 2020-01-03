import torch
from torch_geometric.data import Data


class SimpleBatch(Data):
    r""" A classic batch object wrapper with :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    """

    def __init__(self, batch=None, **kwargs):
        super(SimpleBatch, self).__init__(**kwargs)

        self.batch = batch
        self.__data_class__ = Data

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects. 
        """
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))

        batch_size = data_list[0][keys[0]].size(0)
        for key in keys:
            assert batch_size == data_list[0][key].size(0)

        batch = SimpleBatch()
        batch.__data_class__ = data_list[0].__class__

        for key in keys:
            batch[key] = []

        for i, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                batch[key].append(item)

        for key in batch.keys:
            item = batch[key][0]
            if torch.is_tensor(item) or isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.stack(batch[key])
            else:
                raise ValueError('Unsupported attribute type')

        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1
