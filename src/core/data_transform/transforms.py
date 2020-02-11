from typing import List
import itertools
import numpy as np
import math
import re
import torch
import random
from torch.nn import functional as F
from sklearn.neighbors import NearestNeighbors, KDTree
from torch_geometric.data import Data
from functools import partial
from torch_geometric.nn import fps, radius, knn, voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos, pool_batch
from torch_scatter import scatter_add, scatter_mean

from src.datasets.multiscale_data import MultiScaleData
from src.utils.transform_utils import SamplingStrategy
from src.utils.config import is_list
from torch_geometric.data import Data, Batch
from tqdm import tqdm as tq
from src.utils import is_iterable


class PointCloudFusion(object):

    def _process(self, data_list):
        data = Batch.from_data_list(data_list)
        delattr(data, "batch")
        return data

    def __call__(self, data_list: List[Data]):
        if len(data_list) == 0:
            raise Exception('A list of data should be provided')
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
    r"""Fit the point cloud to a grid and for each point in this grid, 
    create a sphere with a radius r
    Args:
        radius (float or [float] or Tensor): Radius of the sphere to be sampled.
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
                    if self._center and key == 'pos': # Center the sphere.
                        item -= t_center
                    setattr(new_data, key, item)
            new_data.center_label = grid_label
            
            datas.append(new_data)
        return datas       

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in tq(data)]
            data = list(itertools.chain(*data)) # 2d list needs to be flatten
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(radius={}, center={})".format(self.__class__.__name__, self._radius, self._center)

class ComputeKDTree(object):
    r"""Calculate the KDTree and save it within data
    Args:
        leaf_size (float or [float] or Tensor): Depth of the .
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
    r"""Randomly select a sphere of points using a given radius
    Args:
        radius (float or [float] or Tensor): Radius of the sphere to be sampled.
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
                if self._center and key == 'pos': # Center the sphere.
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
        return "{}(radius={}, center={}, sampling_strategy={})".format(self.__class__.__name__, self._radius, self._center, self._sampling_strategy)

class GridSampling(object):
    r"""Clusters points into voxels with size :attr:`size`.
    Args:
        size (float or [float] or Tensor): Size of a voxel (in each dimension).
        start (float or [float] or Tensor, optional): Start coordinates of the
            grid (in each dimension). If set to :obj:`None`, will be set to the
            minimum coordinates found in :obj:`data.pos`.
            (default: :obj:`None`)
        end (float or [float] or Tensor, optional): End coordinates of the grid
            (in each dimension). If set to :obj:`None`, will be set to the
            maximum coordinates found in :obj:`data.pos`.
            (default: :obj:`None`)
        num_classes (int, optional): number of classes in the dataset (speeds up the computation slightly)
    """

    def __init__(self, size, start=None, end=None, num_classes=-1):
        self.size = size
        self.start = start
        self.end = end
        self.num_classes = num_classes

    def _process(self, data):
        num_nodes = data.num_nodes

        if "batch" not in data:
            batch = data.pos.new_zeros(num_nodes, dtype=torch.long)
        else:
            batch = data.batch

        cluster = voxel_grid(data.pos, batch, self.size, self.start, self.end)
        cluster, perm = consecutive_cluster(cluster)

        for key, item in data:
            if bool(re.search("edge", key)):
                raise ValueError("GridSampling does not support coarsening of edges")

            if torch.is_tensor(item) and item.size(0) == num_nodes:
                if key == "y":
                    item = F.one_hot(item, num_classes=self.num_classes)
                    item = scatter_add(item, cluster, dim=0)
                    data[key] = item.argmax(dim=-1)
                elif key == "batch":
                    data[key] = item[perm]
                else:
                    data[key] = scatter_mean(item, cluster, dim=0)        
        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(size={})".format(self.__class__.__name__, self.size)


class RandomSymmetry(object):
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
    def __init__(self, sigma=0.01, clip=0.05):
        """
        simple isotropic additive gaussian noise
        """
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data):
        noise = self.sigma * torch.randn(data.pos.shape)
        noise = noise.clamp(-self.clip, self.clip)
        data.pos = data.pos + noise
        return data

    def __repr__(self):
        return "Random noise of sigma={}".format(self.sigma)


class RandomScaleAnisotropic:
    r"""Scales node positions by a randomly sampled factor :math:`s1, s2, s3` within a
    given interval, *e.g.*, resulting in the transformation matrix

    .. math::
        \begin{bmatrix}
            s1 & 0 & 0 \\
            0 & s2 & 0 \\
            0 & 0 & s3 \\
        \end{bmatrix}

    for three-dimensional positions.

    Args:
        scales (tuple): scaling factor interval, e.g. :obj:`(a, b)`, then scale
            is randomly sampled from the range
            :math:`a \leq \mathrm{scale} \leq b`.
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
    """
    """
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


class RandomRotation(object):
    def __init__(self, mode="vertical"):
        """
        random rotation: either
        """
        self.mode = mode

    def __call__(self, data):

        theta = torch.zeros(3)
        if self.mode == "vertical":
            theta[2] = torch.rand(1) * 2 * torch.tensor(math.pi)
        elif self.mode == "all":
            theta = torch.rand(3) * 2 * torch.tensor(math.pi)
        else:
            raise NotImplementedError("this kind of rotation ({}) " "is not yet available".format(self.mode))
        R = euler_angles_to_rotation_matrix(theta)
        data.pos = torch.mm(data.pos, R.t())
        return data

    def __repr__(self):
        return "Random rotation of mode {}".format(self.mode)


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
    """ Pre-computes a sequence of downsampling / neighboorhood search
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
