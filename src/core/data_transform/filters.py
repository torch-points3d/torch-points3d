import torch

from src.core.data_transform.features import PCACompute, compute_planarity


class FCompose(object):
    """
    """
    def __init__(self, list_filter):
        self.list_filter = list_filter

    def __call__(self, data):

        res = True
        for filter_fn in self.list_filter:
            res = (res and filter_fn(data))
        return res


class PlanarityFilter(object):
    """
    compute planarity and return false if the planarity is above a threshold
    """

    def __init__(self, thresh=0.3):
        self.thresh = thresh

    def __call__(self, data):
        if(getattr(data, 'eigenvalues', None) is None):
            data = PCACompute()(data)
        planarity = compute_planarity(data.eigenvalues)
        return planarity < self.thresh
