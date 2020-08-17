import torch_geometric.transforms as T
from torch_geometric.data import Data

class MultiScaleFixedPoints(object):
    """Randomly samples a fixed number of points over multiple scales and sets them as attributes.

    Parameters
    ----------
    scales: list[int]
        A list of how many points to sample at each scale
    replace: bool
        Pass through to torch_geometric.FixedPoints `replace` parameter.
    allow_duplicates: bool
        Pass through to torch_geometric.FixedPoints `allow_duplicates` parameter.
    allow_duplicates: bool
        Removes all non-scale attributes from the data
    """

    def __init__(self, scales, replace=True, allow_duplicates=False, remove_attrs=True):
        self._scales = scales
        self._replace = replace
        self._allow_duplicates = allow_duplicates
        self._remove_attrs = remove_attrs

    def _process(self, data):
        if self._remove_attrs:
            data = Data(pos=data.pos)

        for scale in self._scales:
            scale_data = Data(pos=data.pos.clone())
            fixed_points = T.FixedPoints(num=scale, replace=self._replace, allow_duplicates=self._allow_duplicates)
            scale_points = fixed_points(scale_data)
            data["scale_" + str(scale)] = scale_points
            
        if self._remove_attrs:
            delattr(data, "pos")
        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(scales={}, replace={}, allow_duplicates={})".format(
            self.__class__.__name__, self._scales, self._replace, self._allow_duplicates
        )
