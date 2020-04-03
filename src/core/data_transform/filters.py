import numpy as np
import torch
import random
from src.core.data_transform.features import PCACompute, compute_planarity


class FCompose(object):
    """
    allow to compose different filters using the boolean operation

    Parameter
    ---------
    list_filter: list
        list of different filter functions we want to apply
    boolean_operation: function, optional
        boolean function to compose the filter (take a pair and return a boolean)
    """
    def __init__(self, list_filter, boolean_operation=np.logical_and):
        self.list_filter = list_filter
        self.boolean_operation = boolean_operation

    def __call__(self, data):

        res = True
        for filter_fn in self.list_filter:
            res = self.boolean_operation(res, filter_fn(data))
        return res


class PlanarityFilter(object):
    """
    compute planarity and return false if the planarity of a pointcloud is above a threshold
    Parameter
    ---------
    thresh: float, optional
        threshold to filter low planar pointcloud
    """

    def __init__(self, thresh=0.3):
        self.thresh = thresh

    def __call__(self, data):
        if(getattr(data, 'eigenvalues', None) is None):
            data = PCACompute()(data)
        planarity = compute_planarity(data.eigenvalues)
        return planarity < self.thresh


class RandomFilter(object):
    """
    Randomly select an elem of the dataset (to have smaller dataset) with a bernouilli distribution of parameter thresh.

    Parameter
    ---------
    thresh: float, optional
        the parameter of the bernouilli function
    """

    def __init__(self, thresh=0.3):
        self.thresh = thresh

    def __call__(self, data):
        return random.random() < self.thresh
