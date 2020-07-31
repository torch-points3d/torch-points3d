from typing import List
import itertools
import numpy as np
import math
import re
import torch
import random
from tqdm.auto import tqdm as tq
from sklearn.neighbors import KDTree
from functools import partial
from torch.nn import functional as F
from torch_geometric.nn.pool.pool import pool_pos, pool_batch
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.transforms import FixedPoints as FP
from torch_points_kernels.points_cpu import ball_query
import numba

from torch_points3d.datasets.multiscale_data import MultiScaleData
from torch_points3d.datasets.registration.pair import Pair
from torch_points3d.utils.transform_utils import SamplingStrategy
from torch_points3d.utils.config import is_list
from torch_points3d.utils import is_iterable
from .grid_transform import group_data, GridSampling3D, shuffle_data
from .features import Random3AxisRotation



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
        if len(data_list) == 0:
            return Data()
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
        Grid_size to be used with GridSampling3D to select spheres center. If None, radius will be used
    delattr_kd_tree: bool, optional
        If True, KDTREE_KEY should be deleted as an attribute if it exists
    center: bool, optional
        If True, a centre transform is apply on each sphere.
    """

    KDTREE_KEY = "kd_tree"

    def __init__(self, radius, grid_size=None, delattr_kd_tree=True, center=True):
        self._radius = eval(radius) if isinstance(radius, str) else float(radius)
        grid_size = eval(grid_size) if isinstance(grid_size, str) else float(grid_size)
        self._grid_sampling = GridSampling3D(size=grid_size if grid_size else self._radius)
        self._delattr_kd_tree = delattr_kd_tree
        self._center = center

    def _process(self, data):
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
            ind = torch.LongTensor(tree.query_radius(pts, r=self._radius)[0])
            sampler = SphereSampling(self._radius, grid_center, align_origin=self._center)
            new_data = sampler(data)
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
    center: bool
        if True then the sphere will be moved to the origin
    """

    def __init__(self, radius, strategy="random", class_weight_method="sqrt", center=True):
        self._radius = eval(radius) if isinstance(radius, str) else float(radius)
        self._sampling_strategy = SamplingStrategy(strategy=strategy, class_weight_method=class_weight_method)
        self._center = center

    def _process(self, data):
        # apply sampling strategy
        random_center = self._sampling_strategy(data)
        random_center = np.asarray(data.pos[random_center])[np.newaxis]
        sphere_sampling = SphereSampling(self._radius, random_center, align_origin=self._center)
        return sphere_sampling(data)

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


class SphereSampling:
    """ Samples points within a sphere

    Parameters
    ----------
    radius : float
        Radius of the sphere
    sphere_centre : torch.Tensor or np.array
        Centre of the sphere (1D array that contains (x,y,z))
    align_origin : bool, optional
        move resulting point cloud to origin
    """

    KDTREE_KEY = "kd_tree"

    def __init__(self, radius, sphere_centre, align_origin=True):
        self._radius = radius
        self._centre = np.asarray(sphere_centre)
        if len(self._centre.shape) == 1:
            self._centre = np.expand_dims(self._centre, 0)
        self._align_origin = align_origin

    def __call__(self, data):
        num_points = data.pos.shape[0]
        if not hasattr(data, self.KDTREE_KEY):
            tree = KDTree(np.asarray(data.pos), leaf_size=50)
            setattr(data, self.KDTREE_KEY, tree)
        else:
            tree = getattr(data, self.KDTREE_KEY)

        t_center = torch.FloatTensor(self._centre)
        ind = torch.LongTensor(tree.query_radius(self._centre, r=self._radius)[0])
        new_data = Data()
        for key in set(data.keys):
            if key == self.KDTREE_KEY:
                continue
            item = data[key]
            if torch.is_tensor(item) and num_points == item.shape[0]:
                item = item[ind]
                if self._align_origin and key == "pos":  # Center the sphere.
                    item -= t_center
            elif torch.is_tensor(item):
                item = item.clone()
            setattr(new_data, key, item)
        return new_data

    def __repr__(self):
        return "{}(radius={}, center={}, align_origin={})".format(
            self.__class__.__name__, self._radius, self._centre, self._align_origin
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


class ScalePos:
    def __init__(self, scale=None):
        self.scale = scale

    def __call__(self, data):
        data.pos *= self.scale
        return data

    def __repr__(self):
        return "{}(scale={})".format(self.__class__.__name__, self.scale)


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
        if(data.norm is not None):
            data.norm = data.norm / scale
            data.norm = torch.nn.functional.normalize(data.norm, dim=1)
        return data

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.scales)


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
        precomputed = [Data(pos=data.pos)]
        upsample = []
        upsample_index = 0
        for index in range(self.num_layers):
            sampler, neighbour_finder = self.strategies["sampler"][index], self.strategies["neighbour_finder"][index]
            support = precomputed[index]
            new_data = Data(pos=support.pos)
            if sampler:
                query = sampler(new_data.clone())
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
                query = new_data

            s_pos, q_pos = support.pos, query.pos
            if hasattr(query, "batch"):
                s_batch, q_batch = support.batch, query.batch
            else:
                s_batch, q_batch = (
                    torch.zeros((s_pos.shape[0]), dtype=torch.long),
                    torch.zeros((q_pos.shape[0]), dtype=torch.long),
                )

            idx_neighboors = neighbour_finder(s_pos, q_pos, batch_x=s_batch, batch_y=q_batch)
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
        (as defined in torch_points3d/datasets/registration/pair.py)
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
            if not hasattr(data, "coords"):
                raise Exception("should quantize first using GridSampling3D")

            if not isinstance(data.coords, torch.IntTensor):
                raise Exception("The pos are expected to be coordinates, so torch.IntTensor")
            data.coords[:, :3] += (torch.rand(3) * 100).type_as(data.coords)
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

    def __init__(self, dropout_ratio: float = 0.2, dropout_application_ratio: float = 0.5):
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


def apply_mask(data, mask, skip_keys=[]):
    size_pos = len(data.pos)
    for k in data.keys:
        if(size_pos == len(data[k]) and k not in skip_keys):
            data[k] = data[k][mask]
    return data

@numba.jit(nopython=True, cache=True)
def rw_mask(pos, ind, dist, mask_vertices, random_ratio=0.04, num_iter=5000):
    rand_ind = np.random.randint(0, len(pos))
    for _ in range(num_iter):
        mask_vertices[rand_ind] = False
        if np.random.rand() < random_ratio:
            rand_ind = np.random.randint(0, len(pos))
        else:
            neighbors = ind[rand_ind][dist[rand_ind] > 0]
            if(len(neighbors) == 0):
                rand_ind = np.random.randint(0, len(pos))
            else:
                n_i = np.random.randint(0, len(neighbors))
                rand_ind = neighbors[n_i]
    return mask_vertices


class RandomWalkDropout(object):
    """
    randomly drop points from input data using random walk

    Parameters
    ----------
    dropout_ratio: float, optional
        Ratio that gets dropped
    num_iter: int, optional
        number of iterations
    radius: float, optional
        radius of the neighborhood search to create the graph
    max_num: int optional
       max number of neighbors
    skip_keys: List optional
        skip_keys where we don't apply the mask
    """

    def __init__(self, dropout_ratio: float = 0.05, num_iter: int = 5000,
                 radius: float = 0.5, max_num: int =- 1, skip_keys: List = []):
        self.dropout_ratio = dropout_ratio
        self.num_iter = num_iter
        self.radius = radius
        self.max_num = max_num
        self.skip_keys = skip_keys

    def __call__(self, data):

        pos = data.pos.detach().cpu().numpy()
        ind, dist = ball_query(data.pos, data.pos,
                               radius=self.radius,
                               max_num=self.max_num, mode=0)
        mask = np.ones(len(pos), dtype=bool)
        mask = rw_mask(pos,
                       ind.detach().cpu().numpy(),
                       dist.detach().cpu().numpy(),
                       mask,
                       num_iter=self.num_iter,
                       random_ratio=self.dropout_ratio)

        data = apply_mask(data, mask, self.skip_keys)

        return data

    def __repr__(self):
        return "{}(dropout_ratio={}, num_iter={}, radius={}, max_num={}, skip_keys={})".format(self.__class__.__name__, self.dropout_ratio, self.num_iter, self.radius, self.max_num, self.skip_keys)


class SphereDropout(object):
    """
    drop out of points on random spheres of fixed radius.
    This function takes n random balls of fixed radius r and drop
    out points inside these balls.

    Parameters
    ----------
    num_sphere: int, optional
        number of random spheres
    radius: float, optional
        radius of the spheres
    """
    def __init__(self, num_sphere: int = 10,
                 radius: float = 5):
        self.num_sphere = num_sphere
        self.radius = radius

    def __call__(self, data):

        pos = data.pos
        list_ind = torch.randint(0, len(pos), (self.num_sphere, ))

        ind, dist = ball_query(data.pos, data.pos[list_ind],
                               radius=self.radius,
                               max_num=-1, mode=1)
        ind = ind[dist[:, 0] > 0]
        mask = torch.ones(len(pos), dtype=torch.bool)
        mask[ind[:, 0]] = False
        data = apply_mask(data, mask)

        return data

    def __repr__(self):
        return "{}(num_sphere={}, radius={})".format(
            self.__class__.__name__, self.num_sphere, self.radius)


class SphereCrop(object):
    """
    crop the point cloud on a sphere. this function.
    takes a ball of radius radius centered on a random point and points
    outside the ball are rejected.

    Parameters
    ----------
    radius: float, optional
        radius of the sphere
    """

    def __init__(self, radius: float = 50):
        self.radius = radius

    def __call__(self, data):
        i = torch.randint(0, len(data.pos), (1, ))
        ind, dist = ball_query(data.pos,
                               data.pos[i].view(1, 3),
                               radius=self.radius,
                               max_num=-1, mode=1)
        ind = ind[dist[:, 0] > 0]
        size_pos = len(data.pos)
        for k in data.keys:
            if(size_pos == len(data[k])):
                data[k] = data[k][ind[:, 0]]
        return data

    def __repr__(self):
        return "{}(radius={})".format(
            self.__class__.__name__, self.radius)


class CubeCrop(object):
    """
    Crop cubically the point cloud. This function take a cube of size c
    centered on a random point, then points outside the cube are rejected.

    Parameters
    ----------
    c: float, optional
        half size of the cube
    rot_x: float_otional
        rotation of the cube around x axis
    rot_y: float_otional
        rotation of the cube around x axis
    rot_z: float_otional
        rotation of the cube around x axis
    """

    def __init__(self, c: float = 1,
                 rot_x: float = 180,
                 rot_y: float = 180,
                 rot_z: float = 180):
        self.c = c
        self.random_rotation = Random3AxisRotation(
            rot_x=rot_x, rot_y=rot_y, rot_z=rot_z)

    def __call__(self, data):
        data_temp = data.clone()
        i = torch.randint(0, len(data.pos), (1, ))
        center = data_temp.pos[i]
        min_square = center - self.c
        max_square = center + self.c
        data_temp.pos = data_temp.pos - center
        data_temp = self.random_rotation(data_temp)
        data_temp.pos = data_temp.pos + center
        mask = torch.prod((data_temp.pos - min_square)>0, dim=1) * torch.prod(
            (max_square - data_temp.pos) > 0, dim=1)
        mask = mask.to(torch.bool)
        data = apply_mask(data, mask)
        return data

    def __repr__(self):
        return "{}(c={}, rotation={})".format(
            self.__class__.__name__,
            self.c,
            self.random_rotation)


class DensityFilter(object):
    """
    Remove points with a low density(compute the density with a radius search and remove points with)
    a low number of neighbors
    Parameter
    radius_nn: float, optional
        radius for the neighbors search
    min_num: int, otional
        minimum number of neighbors to be dense
    skip_keys: int, otional
        list of attributes of data to skip when we apply the mask
    """

    def __init__(self, radius_nn: float = 0.04,
                 min_num: int = 6, skip_keys: List = []):
        self.radius_nn = radius_nn
        self.min_num = min_num
        self.skip_keys = skip_keys

    def __call__(self, data):

        ind, dist = ball_query(data.pos, data.pos,
                               radius=self.radius_nn,
                               max_num=-1, mode=0)

        mask = ((dist > 0).sum(1) > self.min_num)
        data = apply_mask(data, mask, self.skip_keys)
        return data

    def __repr__(self):
        return "{}(radius_nn={}, min_num={}, skip_keys={})".format(self.__class__.__name__,
                                                                   self.radius_nn,
                                                                   self.min_num,
                                                                   self.skip_keys)
