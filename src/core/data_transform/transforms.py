from typing import List
import itertools
import numpy as np
import math
import re
import torch
import random
from tqdm import tqdm as tq
from sklearn.neighbors import NearestNeighbors, KDTree
from functools import partial
from torch.nn import functional as F
from torch_geometric.nn import fps, radius, knn
from torch_geometric.nn.pool.pool import pool_pos, pool_batch
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.transforms import FixedPoints as FP

from src.datasets.multiscale_data import MultiScaleData
from src.datasets.registration.pair import Pair
from src.utils.transform_utils import SamplingStrategy
from src.utils.config import is_list
from src.utils import is_iterable
from .grid_transform import group_data, GridSampling
from .sparse_transforms import shuffle_data


class RemoveAttributes(object):
    """This transform allows to remove unnecessary attributes from data for optimization purposes

    Parameters
    ----------
    attr_names: list
        Remove the attributes from data using the provided `attr_name` within attr_names
    strict: bool=False
        Wether True, it will raise an execption if the provided attr_name isn t within data keys.
    """

    def __init__(self, attr_names=[], strict=False):
        self._attr_names = attr_names
        self._strict = strict

    def __call__(self, data):
        keys = set(data.keys)
        for attr_name in self._attr_names:
            if attr_name not in keys and self._strict:
                raise Exception("attr_name: {} isn t within keys: {}".format(attr_name, keys))
        for attr_name in self._attr_names:
            delattr(data, attr_name)
        return data

    def __repr__(self):
        return "{}(attr_names={}, strict={})".format(self.__class__.__name__, self._attr_names, self._strict)


class PointCloudFusion(object):

    """This transform is responsible to perform a point cloud fusion from a list of data

    - If a list of data is provided -> Create one Batch object with all data
    - If a list of list of data is provided -> Create a list of fused point cloud
    """

    def _process(self, data_list):
        data = Batch.from_data_list(data_list)
        delattr(data, "batch")
        return data

    def __call__(self, data_list: List[Data]):
        if len(data_list) == 0:
            raise Exception("A list of data should be provided")
        elif len(data_list) == 1:
            return data_list[0]
        else:
            if isinstance(data_list[0], list):
                data = [self._process(d) for d in data_list]
            else:
                data = self._process(data_list)
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class GridSphereSampling(object):
    """Fits the point cloud to a grid and for each point in this grid,
    create a sphere with a radius r

    Parameters
    ----------
    radius: float
        Radius of the sphere to be sampled.
    grid_size: float, optional
        Grid_size to be used with GridSampling to select spheres center. If None, radius will be used
    delattr_kd_tree: bool, optional
        If True, KDTREE_KEY should be deleted as an attribute if it exists
    center: bool, optional
        If True, a centre transform is apply on each sphere.
    """

    KDTREE_KEY = "kd_tree"

    def __init__(self, radius, grid_size=None, delattr_kd_tree=True, center=True):
        self._radius = eval(radius) if isinstance(radius, str) else float(radius)

        self._grid_sampling = GridSampling(size=grid_size if grid_size else self._radius)
        self._delattr_kd_tree = delattr_kd_tree
        self._center = center

    def _process(self, data):
        num_points = data.pos.shape[0]

        if not hasattr(data, self.KDTREE_KEY):
            tree = KDTree(np.asarray(data.pos), leaf_size=50)
        else:
            tree = getattr(data, self.KDTREE_KEY)

        # The kdtree has bee attached to data for optimization reason.
        # However, it won't be used for down the transform pipeline and should be removed before any collate func call.
        if hasattr(data, self.KDTREE_KEY) and self._delattr_kd_tree:
            delattr(data, self.KDTREE_KEY)

        # apply grid sampling
        grid_data = self._grid_sampling(data.clone())

        datas = []
        for grid_center in np.asarray(grid_data.pos):
            pts = np.asarray(grid_center)[np.newaxis]

            # Find closest point within the original data
            ind = torch.LongTensor(tree.query(pts, k=1)[1][0])
            grid_label = data.y[ind]

            # Find neighbours within the original data
            t_center = torch.FloatTensor(grid_center)
            ind = torch.LongTensor(tree.query_radius(pts, r=self._radius)[0])

            # Create a new data holder.
            new_data = Data()
            for key in set(data.keys):
                item = data[key].clone()
                if num_points == item.shape[0]:
                    item = item[ind]
                    if self._center and key == "pos":  # Center the sphere.
                        item -= t_center
                    setattr(new_data, key, item)
            new_data.center_label = grid_label

            datas.append(new_data)
        return datas

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in tq(data)]
            data = list(itertools.chain(*data))  # 2d list needs to be flatten
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(radius={}, center={})".format(self.__class__.__name__, self._radius, self._center)


class ComputeKDTree(object):
    """Calculate the KDTree and saves it within data

    Parameters
    -----------
    leaf_size:int
        Size of the leaf node.
    """

    def __init__(self, leaf_size):
        self._leaf_size = leaf_size

    def _process(self, data):
        data.kd_tree = KDTree(np.asarray(data.pos), leaf_size=self._leaf_size)
        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(leaf_size={})".format(self.__class__.__name__, self._leaf_size)


class RandomSphere(object):
    """Select points within a sphere of a given radius. The centre is chosen randomly within the point cloud.

    Parameters
    ----------
    radius: float
        Radius of the sphere to be sampled.
    strategy: str
        choose between `random` and `freq_class_based`. The `freq_class_based` \
        favors points with low frequency class. This can be used to balance unbalanced datasets
    """

    KDTREE_KEY = "kd_tree"

    def __init__(self, radius, strategy="random", class_weight_method="sqrt", delattr_kd_tree=True, center=True):
        self._radius = eval(radius) if isinstance(radius, str) else float(radius)

        self._sampling_strategy = SamplingStrategy(strategy=strategy, class_weight_method=class_weight_method)

        self._delattr_kd_tree = delattr_kd_tree
        self._center = center

    def _process(self, data):
        num_points = data.pos.shape[0]

        if not hasattr(data, self.KDTREE_KEY):
            tree = KDTree(np.asarray(data.pos), leaf_size=50)
        else:
            tree = getattr(data, self.KDTREE_KEY)

        # The kdtree has bee attached to data for optimization reason.
        # However, it won't be used for down the transform pipeline and should be removed before any collate func call.
        if hasattr(data, self.KDTREE_KEY) and self._delattr_kd_tree:
            delattr(data, self.KDTREE_KEY)

        # apply sampling strategy
        random_center = self._sampling_strategy(data)

        center = np.asarray(data.pos[random_center])[np.newaxis]
        t_center = torch.FloatTensor(center)
        ind = torch.LongTensor(tree.query_radius(center, r=self._radius)[0])
        for key in set(data.keys):
            item = data[key]
            if num_points == item.shape[0]:
                item = item[ind]
                if self._center and key == "pos":  # Center the sphere.
                    item -= t_center
                setattr(data, key, item)
        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(radius={}, center={}, sampling_strategy={})".format(
            self.__class__.__name__, self._radius, self._center, self._sampling_strategy
        )


class RandomSymmetry(object):
    """ Apply a random symmetry transformation on the data

    Parameters
    ----------
    axis: Tuple[bool,bool,bool], optional
        axis along which the symmetry is applied
    """

    def __init__(self, axis=[False, False, False]):
        self.axis = axis

    def __call__(self, data):

        for i, ax in enumerate(self.axis):
            if ax:
                if torch.rand(1) < 0.5:
                    data.pos[:, i] *= -1
        return data

    def __repr__(self):
        return "Random symmetry of axes: x={}, y={}, z={}".format(*self.axis)


class RandomNoise(object):
    """ Simple isotropic additive gaussian noise (Jitter)

    Parameters
    ----------
    sigma:
        Variance of the noise
    clip:
        Maximum amplitude of the noise
    """

    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data):
        noise = self.sigma * torch.randn(data.pos.shape)
        noise = noise.clamp(-self.clip, self.clip)
        data.pos = data.pos + noise
        return data

    def __repr__(self):
        return "{}(sigma={}, clip={})".format(self.__class__.__name__, self.sigma, self.clip)


class RandomScaleAnisotropic:
    r""" Scales node positions by a randomly sampled factor ``s1, s2, s3`` within a
    given interval, *e.g.*, resulting in the transformation matrix

    .. math::
        \left[
        \begin{array}{ccc}
            s1 & 0 & 0 \\
            0 & s2 & 0 \\
            0 & 0 & s3 \\
        \end{array}
        \right]


    for three-dimensional positions.

    Parameter
    ---------
    scales:
        scaling factor interval, e.g. ``(a, b)``, then scale \
        is randomly sampled from the range \
        ``a <=  b``. \
    """

    def __init__(self, scales=None, anisotropic=True):
        assert is_iterable(scales) and len(scales) == 2
        assert scales[0] <= scales[1]
        self.scales = scales

    def __call__(self, data):
        scale = self.scales[0] + torch.rand((3,)) * (self.scales[1] - self.scales[0])
        data.pos = data.pos * scale
        return data

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.scales)


def euler_angles_to_rotation_matrix(theta):
    R_x = torch.tensor(
        [[1, 0, 0], [0, torch.cos(theta[0]), -torch.sin(theta[0])], [0, torch.sin(theta[0]), torch.cos(theta[0])]]
    )

    R_y = torch.tensor(
        [[torch.cos(theta[1]), 0, torch.sin(theta[1])], [0, 1, 0], [-torch.sin(theta[1]), 0, torch.cos(theta[1])]]
    )

    R_z = torch.tensor(
        [[torch.cos(theta[2]), -torch.sin(theta[2]), 0], [torch.sin(theta[2]), torch.cos(theta[2]), 0], [0, 0, 1]]
    )

    R = torch.mm(R_z, torch.mm(R_y, R_x))
    return R


class MeshToNormal(object):
    """ Computes mesh normals (IN PROGRESS)
    """

    def __init__(self):
        pass

    def __call__(self, data):
        if hasattr(data, "face"):
            pos = data.pos
            face = data.face
            vertices = [pos[f] for f in face]
            normals = torch.cross(vertices[0] - vertices[1], vertices[0] - vertices[2], dim=1)
            normals = F.normalize(normals)
            data.normals = normals
        return data

    def __repr__(self):
        return "{}".format(self.__class__.__name__)


class MultiScaleTransform(object):
    """ Pre-computes a sequence of downsampling / neighboorhood search on the CPU.
    This currently only works on PARTIAL_DENSE formats

    Parameters
    -----------
    strategies: Dict[str, object]
        Dictionary that contains the samplers and neighbour_finder
    """

    def __init__(self, strategies):
        self.strategies = strategies
        self.num_layers = len(self.strategies["sampler"])

    @staticmethod
    def __inc__wrapper(func, special_params):
        def new__inc__(key, num_nodes, special_params=None, func=None):
            if key in special_params:
                return special_params[key]
            else:
                return func(key, num_nodes)

        return partial(new__inc__, special_params=special_params, func=func)

    def __call__(self, data: Data) -> MultiScaleData:
        # Compute sequentially multi_scale indexes on cpu
        data.contiguous()
        ms_data = MultiScaleData.from_data(data)
        precomputed = [data]
        upsample = []
        upsample_index = 0
        for index in range(self.num_layers):
            sampler, neighbour_finder = self.strategies["sampler"][index], self.strategies["neighbour_finder"][index]
            support = precomputed[index]
            if sampler:
                query = sampler(support.clone())
                query.contiguous()

                if len(self.strategies["upsample_op"]):
                    if upsample_index >= len(self.strategies["upsample_op"]):
                        raise ValueError("You are missing some upsample blocks in your network")

                    upsampler = self.strategies["upsample_op"][upsample_index]
                    upsample_index += 1
                    pre_up = upsampler.precompute(query, support)
                    upsample.append(pre_up)
                    special_params = {}
                    special_params["x_idx"] = query.num_nodes
                    special_params["y_idx"] = support.num_nodes
                    setattr(pre_up, "__inc__", self.__inc__wrapper(pre_up.__inc__, special_params))
            else:
                query = support.clone()

            s_pos, q_pos = support.pos, query.pos
            if hasattr(query, "batch"):
                s_batch, q_batch = support.batch, query.batch
            else:
                s_batch, q_batch = (
                    torch.zeros((s_pos.shape[0]), dtype=torch.long),
                    torch.zeros((q_pos.shape[0]), dtype=torch.long),
                )

            idx_neighboors, _ = neighbour_finder(s_pos, q_pos, batch_x=s_batch, batch_y=q_batch)
            special_params = {}
            special_params["idx_neighboors"] = s_pos.shape[0]
            setattr(query, "idx_neighboors", idx_neighboors)
            setattr(query, "__inc__", self.__inc__wrapper(query.__inc__, special_params))
            precomputed.append(query)
        ms_data.multiscale = precomputed[1:]
        upsample.reverse()  # Switch to inner layer first
        ms_data.upsample = upsample
        return ms_data

    def __repr__(self):
        return "{}".format(self.__class__.__name__)


class ShuffleData(object):
    """ This transform allow to shuffle feature, pos and label tensors within data
    """

    def _process(self, data):
        return shuffle_data(data)

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in tq(data)]
            data = list(itertools.chain(*data))  # 2d list needs to be flatten
        else:
            data = self._process(data)
        return data


class PairTransform(object):
    def __init__(self, transform):
        """
        apply the transform for a pair of data
        (as defined in src/datasets/registration/pair.py)
        """
        self.transform = transform

    def __call__(self, data):
        data_source, data_target = data.to_data()
        data_source = self.transform(data_source)
        data_target = self.transform(data_target)
        return Pair.make_pair(data_source, data_target)

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class ShiftVoxels:
    """ Trick to make Sparse conv invariant to even and odds coordinates
    https://github.com/chrischoy/SpatioTemporalSegmentation/blob/master/lib/train.py#L78

    Parameters
    -----------
    apply_shift: bool:
        Whether to apply the shift on indices
    """

    def __init__(self, apply_shift=True):
        self._apply_shift = apply_shift

    def __call__(self, data):
        if self._apply_shift:
            if not isinstance(data.pos, torch.IntTensor):
                raise Exception("The pos are expected to be coordinates, so torch.IntTensor")
            data.pos[:, :3] += (torch.rand(3) * 100).type_as(data.pos)
        return data

    def __repr__(self):
        return "{}(apply_shift={})".format(self.__class__.__name__, self._apply_shift)


class RandomDropout:
    """ Randomly drop points from the input data

    Parameters
    ----------
    dropout_ratio : float, optional
        Ratio that gets dropped
    dropout_application_ratio   : float, optional
        chances of the dropout to be applied
    """

    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data):
        if random.random() < self.dropout_application_ratio:
            N = len(data.pos)
            data = FP(int(N * (1 - self.dropout_ratio)))(data)
        return data

    def __repr__(self):
        return "{}(dropout_ratio={}, dropout_application_ratio={})".format(
            self.__class__.__name__, self.dropout_ratio, self.dropout_application_ratio
        )

