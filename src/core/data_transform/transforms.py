from typing import List
import itertools
import numpy as np
import math
import re
import torch
import random
from torch.nn import functional as F
from sklearn.neighbors import NearestNeighbors, KDTree
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
from src.modules.MinkowskiEngine import to_sparse_input, remove_duplicates_func, shuffle_data

class RemoveAttributes(object):
    """[This transform allows to remove unnecessary attributes from data for optimization purposes]
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

class ShuffleData(object):
    """[This transform allows to shuffle the data]
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

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)

class RemoveDuplicatedCoords(object):
    """[This transform allows to remove duplicated indices within data for sparse input]
    """

    def __init__(self, shuffle=False):
        self._shuffle = shuffle
        if self._shuffle:
            self._shuffle_transform = ShuffleData()

    def _process(self, data):
        if self._shuffle:
            data = self._shuffle_transform(data)
        return remove_duplicates_func(data)

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in tq(data)]
            data = list(itertools.chain(*data))  # 2d list needs to be flatten
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(shuffle={})".format(self.__class__.__name__, self._shuffle)


class ToSparseInput(object):
    """[This transform allows to prepare data for sparse model as SparseConv / Minkowski Engine]
    """

    def __init__(self, grid_size=None, save_delta: bool=False, save_delta_norm:bool=False, remove_duplicates:bool=True, apply_mean:bool=True):
        if grid_size is None:
            raise Exception("Grid size should be provided")

        elif grid_size == 0:
            raise Exception("Grid size should not be equal to 0")

        self._grid_size = grid_size
        self._save_delta = save_delta
        self._save_delta_norm = save_delta_norm
        self._remove_duplicates = remove_duplicates
        self._apply_mean = apply_mean

    def _process(self, data):
        return to_sparse_input(data, self._grid_size, save_delta=self._save_delta, remove_duplicates=self._remove_duplicates, apply_mean=self._apply_mean)

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in tq(data)]
            data = list(itertools.chain(*data))  # 2d list needs to be flatten
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(grid_size={}, save_delta={}, save_delta_norm={}, remove_duplicates={}, apply_mean={})".format(self.__class__.__name__, self._grid_size, self._save_delta, self._save_delta_norm, self._remove_duplicates, self._apply_mean)


class PointCloudFusion(object):
    r"""This transform is responsible to perform a point cloud fusion from a list of data
    If a list of data is provided -> Create one Batch object with all data
    If a list of list of data is provided -> Create a list of fused point cloud
    Args:
        radius (float or [float] or Tensor): Radius of the sphere to be sampled.
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


class GridSampling(object):
    """ Clusters points into voxels with size :attr:`size`.

    Parameters
    ----------
    size: float
        Size of a voxel (in each dimension).
    start: float
        Start coordinates of the grid (in each dimension). \
        If set to `None`, will be set to the minimum coordinates found in `data.pos`. (default: `None`)
    end: float
        End coordinates of the grid (in each dimension). \
        If set to `None`, will be set to the maximum coordinates found in `data.pos`. (default: `None`)
    num_classes: max number of classes for one hot encoding of y vector
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
                elif key == "batch" or key == SaveOriginalPosId.KEY:
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
        return "Random noise of sigma={}".format(self.sigma)


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

class AddFeatsByKeys(object):
    """[This transform takes a list of attributes names and if allowed, add them to x]
    
    Example:

        Before calling "AddFeatsByKeys", if data.x was empty

        - transform: AddFeatsByKeys
          params:
              list_add_to_x: [False, True, True]    
              feat_names: ['normal', 'rgb', "elevation"]
              input_nc_feats: [3, 3, 1]

        After calling "AddFeatsByKeys", data.x contains "rgb" and "elevation". Its shape[-1] == 4 (rgb:3 + elevation:1)
    """

    def __init__(self, list_add_to_x: List[bool], feat_names: List[str], input_nc_feats: List[int] = None, stricts: List[bool] = None):
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

        transforms = [AddFeatByKey(add_to_x, feat_name, input_nc_feat=input_nc_feat, strict=strict) for add_to_x, feat_name, input_nc_feat, strict in zip(list_add_to_x, feat_names, input_nc_feats, stricts)]

        self.transform = Compose(transforms)

    def __call__(self, data):
        return self.transform(data)


class AddFeatByKey(object):
    """This transform is responsible to get an attribute under feat_name and add it to x if add_to_x is True
    
    Paremeters
    ----------
    add_to_x: bool
        Control if the feature is going to be added/concatenated to x
    feat_name: str
        The feature to be found within data to be added/concatenated to x
    
    input_nc_feat: int, optional
        If provided, check if dimension feature check last dimension (default: ``None``)
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
                data.x = feat
            else:
                if x.shape[0] == feat.shape[0]:
                    if x.dim() == 1:
                        x = x.unsqueeze(-1)
                    if feat.dim() == 1:
                        feat = feat.unsqueeze(-1)
                    data.x = torch.cat([x, feat], axis=-1)
                else:
                    raise Exception("The tensor x and {} can't be concatenated, x: {}, feat: {}".format(self._feat_name, 
                                                                                                        x.pos.shape[0], 
                                                                                                        feat.pos.shape[0]))
        return data

    def __repr__(self):
        return  "{}(add_to_x: {}, feat_name: {}, strict: {})".format(self.__class__.__name__, self._add_to_x, self._feat_name, self._strict)

class SaveOriginalPosId:
    """ Transform that adds the index of the point to the data object
    This allows us to track this point from the output back to the input data object
    """

    KEY = "origin_id"

    def __call__(self, data):
        setattr(data, self.KEY, torch.arange(0, data.pos.shape[0]))
        return data
