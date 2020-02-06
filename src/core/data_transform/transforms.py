import numpy as np
import math
import re
import torch
from torch.nn import functional as F
from sklearn.neighbors import NearestNeighbors, KDTree
from torch_geometric.data import Data
from functools import partial
from torch_geometric.nn import fps, radius, knn, voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos, pool_batch
from torch_scatter import scatter_add, scatter_mean
from src.datasets.multiscale_data import MultiScaleData


class ComputeKDTree(object):
    r"""Calculate the KDTree and save it within data
    Args:
        leaf_size (float or [float] or Tensor): Depth of the .
    """
    def __init__(self, leaf_size):
        self._leaf_size = leaf_size

    def __call__(self, data):
        data.kd_tree = KDTree(np.asarray(data.pos), leaf_size=self._leaf_size)
        return data

    def __repr__(self):
        return "{}(leaf_size={})".format(self.__class__.__name__, self._leaf_size)

class RandomSphere(object):
    r"""Randomly select a sphere of points using a given radius
    Args:
        radius (float or [float] or Tensor): Radius of the sphere to be sampled.
    """
    key = "kd_tree"
    STRATEGIES = ["RANDOM", "FREQ_CLASS_BASED"]
    def __init__(self, radius, strategy="RANDOM", class_weight_method="sqrt"):
        self._radius = eval(radius) if isinstance(radius, str) else float(radius)
        if strategy in self.STRATEGIES:
            self._strategy = strategy
        else:
            raise NotImplementedError('Only the followings strategy are implemented {}'.format(self.STRATEGIES))
        self._class_weight_method = "sqrt" if class_weight_method.lower() == "sqrt" else "None"

    def __call__(self, data):
        if not hasattr(data, self.key):
            tree = KDTree(np.asarray(data.pos), leaf_size=50)
        else:
            tree = getattr(data, self.key)
            delattr(data, self.key)

        num_points = data.pos.shape[0]

        if self._strategy == "RANDOM":
            random_center = np.random.randint(0, len(data.pos))
        
        elif self._strategy == "FREQ_CLASS_BASED": 
            labels = np.asarray(data.y)
            uni, uni_counts = np.unique(np.asarray(data.y), return_counts=True)
            uni_counts = uni_counts.mean() / uni_counts
            if self._class_weight_method == "sqrt":
                uni_counts = np.sqrt(uni_counts)
            uni_counts /= np.sum(uni_counts)
            chosen_label = np.random.choice(uni, p= uni_counts)
            random_center = np.random.choice(np.argwhere(labels == chosen_label).flatten())
        else:
            raise NotImplementedError

        pts = np.asarray(data.pos[random_center])[np.newaxis]
        ind = torch.LongTensor(tree.query_radius(pts, r=self._radius)[0])
        for key in set(data.keys):
            item = data[key]
            if num_points == item.shape[0]:
                setattr(data, key, item[ind])
        return data

    def __repr__(self):
        return "{}(radius={}, strategy={}, class_weight_method={})".format(self.__class__.__name__, self._radius, self._strategy, self._class_weight_method)

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

    def __call__(self, data):
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
        for index in range(self.num_layers):
            sampler, neighbour_finder = self.strategies["sampler"][index], self.strategies["neighbour_finder"][index]
            support = precomputed[index]
            if sampler:
                query = sampler(support.clone())
            else:
                query = support.clone()

            query.contiguous()
            s_pos, q_pos = support.pos, query.pos
            if hasattr(query, "batch"):
                s_batch, q_batch = support.batch, query.batch
            else:
                s_batch, q_batch = (
                    torch.zeros((s_pos.shape[0]), dtype=torch.long),
                    torch.zeros((q_pos.shape[0]), dtype=torch.long),
                )

            idx_neighboors, _ = neighbour_finder(s_pos, q_pos, batch_x=s_batch, batch_y=q_batch)
            special_params = {"idx_neighboors": s_pos.shape[0]}
            setattr(query, "idx_neighboors", idx_neighboors)
            setattr(query, "__inc__", self.__inc__wrapper(query.__inc__, special_params))
            precomputed.append(query)
        ms_data.multiscale = precomputed[1:]
        return ms_data

    @staticmethod
    def get_tensor_name(index, field):
        return "precomputed_%i_%s" % (index, field)

    def __repr__(self):
        return "{}".format(self.__class__.__name__)
