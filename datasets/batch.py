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

        # Check if all dimensions matches and we can concatenate data
        # if len(data_list) > 0:
        #    for data in data_list[1:]:
        #        for key in keys:
        #            assert data_list[0][key].shape == data[key].shape

        batch = SimpleBatch()
        batch.__data_class__ = data_list[0].__class__

        for key in keys:
            batch[key] = []

        for _, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                batch[key].append(item)

        for key in batch.keys:
            item = batch[key][0]
            if (
                torch.is_tensor(item)
                or isinstance(item, int)
                or isinstance(item, float)
            ):
                batch[key] = torch.stack(batch[key])
            else:
                raise ValueError("Unsupported attribute type")

        return batch.contiguous()
        # return [batch.x.transpose(1, 2).contiguous(), batch.pos, batch.y.view(-1)]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1
