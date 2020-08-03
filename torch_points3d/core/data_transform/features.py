from typing import List, Optional
from tqdm.auto import tqdm as tq
import itertools
import numpy as np
import math
import re
import torch
import random
from torch.nn import functional as F
from functools import partial

from torch_geometric.nn import fps, radius, knn, voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos, pool_batch
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.data import Data, Batch
from torch_points3d.datasets.multiscale_data import MultiScaleData
from torch_points3d.utils.transform_utils import SamplingStrategy
from torch_points3d.utils.config import is_list
from torch_points3d.utils import is_iterable
from torch_points3d.utils.geometry import euler_angles_to_rotation_matrix


class Random3AxisRotation(object):
    """
    Rotate pointcloud with random angles along x, y, z axis

    The angles should be given `in degrees`.

    Parameters
    -----------
    apply_rotation: bool:
        Whether to apply the rotation
    rot_x: float
        Rotation angle in degrees on x axis
    rot_y: float
        Rotation anglei n degrees on y axis
    rot_z: float
        Rotation angle in degrees on z axis
    """

    def __init__(self, apply_rotation: bool = True, rot_x: float = None, rot_y: float = None, rot_z: float = None):
        self._apply_rotation = apply_rotation
        if apply_rotation:
            if (rot_x is None) and (rot_y is None) and (rot_z is None):
                raise Exception("At least one rot_ should be defined")

        self._rot_x = np.abs(rot_x) if rot_x else 0
        self._rot_y = np.abs(rot_y) if rot_y else 0
        self._rot_z = np.abs(rot_z) if rot_z else 0

        self._degree_angles = [self._rot_x, self._rot_y, self._rot_z]

    def generate_random_rotation_matrix(self):
        thetas = []
        for axis_ind, deg_angle in enumerate(self._degree_angles):
            if deg_angle > 0:
                rand_deg_angle = random.random() * deg_angle
                rand_radian_angle = float(rand_deg_angle * np.pi) / 180.0
                thetas.append(torch.tensor(rand_radian_angle))
            else:
                thetas.append(torch.tensor(0.0))
        return euler_angles_to_rotation_matrix(thetas)

    def __call__(self, data):
        if self._apply_rotation:
            pos = data.pos
            M = self.generate_random_rotation_matrix()
            data.pos = pos @ M.T
            if(data.norm is not None):
                data.norm = data.norm @ M.T
        return data

    def __repr__(self):
        return "{}(apply_rotation={}, rot_x={}, rot_y={}, rot_z={})".format(
            self.__class__.__name__, self._apply_rotation, self._rot_x, self._rot_y, self._rot_z
        )


class RandomTranslation(object):
    """
    random translation
    Parameters
    -----------
    delta_min: list
        min translation
    delta_max: list
        max translation
    """
    def __init__(self, delta_max: List = [1.0, 1.0, 1.0],
                 delta_min: List = [-1.0, -1.0, -1.0]):
        self.delta_max = torch.tensor(delta_max)
        self.delta_min = torch.tensor(delta_min)

    def __call__(self, data):
        pos = data.pos
        trans = torch.rand(3) * (self.delta_max - self.delta_min) + self.delta_min
        data.pos = pos + trans
        return data

    def __repr__(self):
        return "{}(delta_min={}, delta_max={})".format(
            self.__class__.__name__, self.delta_min, self.delta_max
        )


class AddFeatsByKeys(object):
    """This transform takes a list of attributes names and if allowed, add them to x

    Example:

        Before calling "AddFeatsByKeys", if data.x was empty

        - transform: AddFeatsByKeys
          params:
              list_add_to_x: [False, True, True]
              feat_names: ['normal', 'rgb', "elevation"]
              input_nc_feats: [3, 3, 1]

        After calling "AddFeatsByKeys", data.x contains "rgb" and "elevation". Its shape[-1] == 4 (rgb:3 + elevation:1)
        If input_nc_feats was [4, 4, 1], it would raise an exception as rgb dimension is only 3.

    Paremeters
    ----------
    list_add_to_x: List[bool]
        For each boolean within list_add_to_x, control if the associated feature is going to be concatenated to x
    feat_names: List[str]
        The list of features within data to be added to x
    input_nc_feats: List[int], optional
        If provided, evaluate the dimension of the associated feature shape[-1] found using feat_names and this provided value. It allows to make sure feature dimension didn't change
    stricts: List[bool], optional
        Recommended to be set to list of True. If True, it will raise an Exception if feat isn't found or dimension doesn t match.
    delete_feats: List[bool], optional
        Wether we want to delete the feature from the data object. List length must match teh number of features added.
    """

    def __init__(
        self,
        list_add_to_x: List[bool],
        feat_names: List[str],
        input_nc_feats: List[Optional[int]] = None,
        stricts: List[bool] = None,
        delete_feats: List[bool] = None,
    ):

        self._feat_names = feat_names
        self._list_add_to_x = list_add_to_x
        self._delete_feats = delete_feats
        if self._delete_feats:
            assert len(self._delete_feats) == len(self._feat_names)
        from torch_geometric.transforms import Compose

        num_names = len(feat_names)
        if num_names == 0:
            raise Exception("Expected to have at least one feat_names")

        assert len(list_add_to_x) == num_names

        if input_nc_feats:
            assert len(input_nc_feats) == num_names
        else:
            input_nc_feats = [None for _ in range(num_names)]

        if stricts:
            assert len(stricts) == num_names
        else:
            stricts = [True for _ in range(num_names)]

        transforms = [
            AddFeatByKey(add_to_x, feat_name, input_nc_feat=input_nc_feat, strict=strict)
            for add_to_x, feat_name, input_nc_feat, strict in zip(list_add_to_x, feat_names, input_nc_feats, stricts)
        ]

        self.transform = Compose(transforms)

    def __call__(self, data):
        data = self.transform(data)
        if self._delete_feats:
            for feat_name, delete_feat in zip(self._feat_names, self._delete_feats):
                if delete_feat:
                    delattr(data, feat_name)
        return data

    def __repr__(self):
        msg = ""
        for f, a in zip(self._feat_names, self._list_add_to_x):
            msg += "{}={}, ".format(f, a)
        return "{}({})".format(self.__class__.__name__, msg[:-2])


class AddFeatByKey(object):
    """This transform is responsible to get an attribute under feat_name and add it to x if add_to_x is True

    Paremeters
    ----------
    add_to_x: bool
        Control if the feature is going to be added/concatenated to x
    feat_name: str
        The feature to be found within data to be added/concatenated to x
    input_nc_feat: int, optional
        If provided, check if feature last dimension maches provided value.
    strict: bool, optional
        Recommended to be set to True. If False, it won't break if feat isn't found or dimension doesn t match. (default: ``True``)
    """

    def __init__(self, add_to_x, feat_name, input_nc_feat=None, strict=True):

        self._add_to_x: bool = add_to_x
        self._feat_name: str = feat_name
        self._input_nc_feat = input_nc_feat
        self._strict: bool = strict

    def __call__(self, data: Data):
        if not self._add_to_x:
            return data
        feat = getattr(data, self._feat_name, None)
        if feat is None:
            if self._strict:
                raise Exception("Data should contain the attribute {}".format(self._feat_name))
            else:
                return data
        else:
            if self._input_nc_feat:
                feat_dim = 1 if feat.dim() == 1 else feat.shape[-1]
                if self._input_nc_feat != feat_dim and self._strict:
                    raise Exception("The shape of feat: {} doesn t match {}".format(feat.shape, self._input_nc_feat))
            x = getattr(data, "x", None)
            if x is None:
                if self._strict and data.pos.shape[0] != feat.shape[0]:
                    raise Exception("We expected to have an attribute x")
                if feat.dim() == 1:
                    feat = feat.unsqueeze(-1)
                data.x = feat
            else:
                if x.shape[0] == feat.shape[0]:
                    if x.dim() == 1:
                        x = x.unsqueeze(-1)
                    if feat.dim() == 1:
                        feat = feat.unsqueeze(-1)
                    data.x = torch.cat([x, feat], dim=-1)
                else:
                    raise Exception(
                        "The tensor x and {} can't be concatenated, x: {}, feat: {}".format(
                            self._feat_name, x.pos.shape[0], feat.pos.shape[0]
                        )
                    )
        return data

    def __repr__(self):
        return "{}(add_to_x: {}, feat_name: {}, strict: {})".format(
            self.__class__.__name__, self._add_to_x, self._feat_name, self._strict
        )


def compute_planarity(eigenvalues):
    r"""
    compute the planarity with respect to the eigenvalues of the covariance matrix of the pointcloud
    let
    :math:`\lambda_1, \lambda_2, \lambda_3` be the eigenvalues st:

    .. math::
        \lambda_1 \leq \lambda_2 \leq \lambda_3

    then planarity is defined as:

    .. math::
        planarity = \frac{\lambda_2 - \lambda_1}{\lambda_3}
    """

    return (eigenvalues[1] - eigenvalues[0]) / eigenvalues[2]


class NormalFeature(object):
    """
    add normal as feature. if it doesn't exist, compute normals
    using PCA
    """

    def __call__(self, data):
        if data.norm is None:
            raise NotImplementedError("TODO: Implement normal computation")

        norm = data.norm
        if data.x is None:
            data.x = norm
        else:
            data.x = torch.cat([data.x, norm], -1)
        return data


class PCACompute(object):
    r"""
    compute `Principal Component Analysis <https://en.wikipedia.org/wiki/Principal_component_analysis>`__ of a point cloud :math:`x_1,\dots, x_n`.
    It computes the eigenvalues and the eigenvectors of the matrix :math:`C` which is the covariance matrix of the point cloud:

    .. math::
        x_{centered} &= \frac{1}{n} \sum_{i=1}^n x_i

        C &= \frac{1}{n} \sum_{i=1}^n (x_i - x_{centered})(x_i - x_{centered})^T

    store the eigen values and the eigenvectors in data.
    in eigenvalues attribute and eigenvectors attributes.
    data.eigenvalues is a tensor :math:`(\lambda_1, \lambda_2, \lambda_3)` such that :math:`\lambda_1 \leq \lambda_2 \leq \lambda_3`.

    data.eigenvectors is a 3 x 3 matrix such that the column are the eigenvectors associated to their eigenvalues
    Therefore, the first column of data.eigenvectors estimates the normal at the center of the pointcloud.
    """

    def __call__(self, data):
        pos_centered = data.pos - data.pos.mean(axis=0)
        cov_matrix = pos_centered.T.mm(pos_centered) / len(pos_centered)
        eig, v = torch.symeig(cov_matrix, eigenvectors=True)
        data.eigenvalues = eig
        data.eigenvectors = v
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class AddOnes(object):
    """
    Add ones tensor to data
    """

    def __call__(self, data):
        num_nodes = data.pos.shape[0]
        data.ones = torch.ones((num_nodes, 1)).float()
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class XYZFeature(object):
    """
    Add the X, Y and Z as a feature
    Parameters
    -----------
    add_x: bool [default: False]
        whether we add the x position or not
    add_y: bool [default: False]
        whether we add the y position or not
    add_z: bool [default: True]
        whether we add the z position or not
    """

    def __init__(self, add_x=False, add_y=False, add_z=True):
        self._axis = []
        axis_names = ["x", "y", "z"]
        if add_x:
            self._axis.append(0)
        if add_y:
            self._axis.append(1)
        if add_z:
            self._axis.append(2)

        self._axis_names = [axis_names[idx_axis] for idx_axis in self._axis]

    def __call__(self, data):
        assert data.pos is not None
        for axis_name, id_axis in zip(self._axis_names, self._axis):
            f = data.pos[:, id_axis].clone()
            setattr(data, "pos_{}".format(axis_name), f)
        return data

    def __repr__(self):
        return "{}(axis={})".format(self.__class__.__name__, self._axis_names)
