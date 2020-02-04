from typing import List, Optional
import torch
import copy
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import Batch


class MultiScaleData(Data):
    def __init__(self, x=None, y=None, pos=None, multiscale: Optional[List[Data]] = None, **kwargs):
        self.x = x
        self.y = y
        self.pos = pos
        self.multiscale = multiscale
        for key, item in kwargs.items():
            self[key] = item

    def apply(self, func, *keys):
        r"""Applies the function :obj:`func` to all tensor and Data attributes
        :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
        all present attributes.
        """
        for key, item in self(*keys):
            if torch.is_tensor(item):
                self[key] = func(item)
        for scale in range(self.num_scales):
            self.multiscale[scale] = self.multiscale[scale].apply(func)
        return self

    @property
    def num_scales(self):
        """ Number of scales in the multiscale array
        """
        return len(self.multiscale) if self.multiscale else 0

    @classmethod
    def from_data(cls, data):
        ms_data = cls()
        for k, item in data:
            ms_data[k] = item
        return ms_data


class MultiScaleBatch(MultiScaleData):
    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`."""
        for data in data_list:
            assert isinstance(data, MultiScaleData)
        num_scales = data_list[0].num_scales
        for data_entry in data_list:
            assert data_entry.num_scales == num_scales, "All data objects should contain the same number of scales"

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert "batch" not in keys

        batch = MultiScaleBatch()
        batch.__data_class__ = data_list[0].__class__
        batch.__slices__ = {key: [0] for key in keys}

        for key in keys:
            batch[key] = []

        for key in follow_batch:
            batch["{}_batch".format(key)] = []

        cumsum = {key: 0 for key in keys}
        batch.batch = []
        for i, data in enumerate(data_list):
            for key in data.keys:
                if key == "multiscale":
                    continue
                item = data[key]
                if torch.is_tensor(item) and item.dtype != torch.bool:
                    item = item + cumsum[key]
                if torch.is_tensor(item):
                    size = item.size(data.__cat_dim__(key, data[key]))
                else:
                    size = 1
                batch.__slices__[key].append(size + batch.__slices__[key][-1])
                cumsum[key] = cumsum[key] + data.__inc__(key, item)
                batch[key].append(item)

                if key in follow_batch:
                    item = torch.full((size,), i, dtype=torch.long)
                    batch["{}_batch".format(key)].append(item)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes,), i, dtype=torch.long)
                batch.batch.append(item)

        if num_nodes is None:
            batch.batch = None

        for key in batch.keys:
            if key == "multiscale":
                continue
            item = batch[key][0]
            if torch.is_tensor(item):
                batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])
            else:
                raise ValueError("Unsupported attribute type")

        batch.multiscale = []
        for scale in range(num_scales):
            ms_scale = []
            for data_entry in data_list:
                ms_scale.append(data_entry.multiscale[scale])
            batch.multiscale.append(Batch.from_data_list(ms_scale))

        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()
