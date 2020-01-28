import torch
import torch_geometric
from torch_geometric.data import Data



class Pair(Data):
    r"""
    A plain old python object modeling a pair of graph or point cloud.
    With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    """

    def __init__(self, **kwargs):
        super(Pair, self).__init__(**kwargs)
        self.__data_class__ = Data
        self.__slices__ = None

    @staticmethod
    def from_pair(data_source, data_target):
        r"""
        construct a Pair object from a pair of data holding class
        :class:`torch_geometric.data.Data` objects.
        """

        pair = Pair()

        for key_source in data_source.keys:
            pair['source_{}'.format(key_source)] = data_source[key_source]
        for key_target in data_target.keys:
            pair['target_{}'.format(key_target)] = data_target[key_target]

        return pair.contiguous()

    def to_pair(self):

        data_source = self.__data_class__()
        data_target = self.__data_class__()

        for key in self.keys:
            elem = key.split('_')
            original_key = ""
            for e in elem[1:]:
                if(len(original_key) > 0):
                    original_key = original_key + "_" + e
                else:
                    original_key = original_key + e

            if(elem[0] == 'source'):
                data_source[original_key] = self[key]
            elif(elem[0] == 'target'):
                data_target[original_key] = self[key]
            else:
                data_source[key] = self[key]
                data_target[key] = self[key]
        return data_source, data_target
