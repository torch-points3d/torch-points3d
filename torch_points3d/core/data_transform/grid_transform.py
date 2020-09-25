from typing import *
import numpy as np
import numpy
import random
import scipy
import re
import torch
import logging
import torch.nn.functional as F
import collections
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn import voxel_grid
from torch_geometric.data import Data
from torch_cluster import grid_cluster

log = logging.getLogger(__name__)


# Label will be the majority label in each voxel
_INTEGER_LABEL_KEYS = ["y", "instance_labels"]


def shuffle_data(data):
    num_points = data.pos.shape[0]
    shuffle_idx = torch.randperm(num_points)
    for key in set(data.keys):
        item = data[key]
        if torch.is_tensor(item) and num_points == item.shape[0]:
            data[key] = item[shuffle_idx]
    return data


def group_data(data, cluster=None, unique_pos_indices=None, mode="last", skip_keys=[]):
    """ Group data based on indices in cluster.
    The option ``mode`` controls how data gets agregated within each cluster.

    Parameters
    ----------
    data : Data
        [description]
    cluster : torch.Tensor
        Tensor of the same size as the number of points in data. Each element is the cluster index of that point.
    unique_pos_indices : torch.tensor
        Tensor containing one index per cluster, this index will be used to select features and labels
    mode : str
        Option to select how the features and labels for each voxel is computed. Can be ``last`` or ``mean``.
        ``last`` selects the last point falling in a voxel as the representent, ``mean`` takes the average.
    skip_keys: list
        Keys of attributes to skip in the grouping
    """

    assert mode in ["mean", "last"]
    if mode == "mean" and cluster is None:
        raise ValueError("In mean mode the cluster argument needs to be specified")
    if mode == "last" and unique_pos_indices is None:
        raise ValueError("In last mode the unique_pos_indices argument needs to be specified")

    num_nodes = data.num_nodes
    for key, item in data:
        if bool(re.search("edge", key)):
            raise ValueError("Edges not supported. Wrong data type.")
        if key in skip_keys:
            continue

        if torch.is_tensor(item) and item.size(0) == num_nodes:
            # if key in _INTEGER_LABEL_KEYS:
            #     data[key] = item[unique_pos_indices]
            #     item_min = item.min()
            #     item = F.one_hot(item - item_min)
            #     item = scatter_add(item, cluster, dim=0)
            #     flatten_labels_voxels = torch.nonzero(item, as_tuple=True)[0]
            #     mutli_label_voxels = torch.unique(flatten_labels_voxels, return_counts=True)[1] > 1
            #     data[key][mutli_label_voxels] = -1
            if mode == "last" or key == "batch" or key == SaveOriginalPosId.KEY:
                data[key] = item[unique_pos_indices]
            elif mode == "mean":
                is_item_bool = item.dtype == torch.bool
                if is_item_bool:
                    item = item.int()
                if key in _INTEGER_LABEL_KEYS:
                    item_min = item.min()
                    item = F.one_hot(item - item_min)
                    item = scatter_add(item, cluster, dim=0)
                    data[key] = item.argmax(dim=-1) + item_min
                else:
                    data[key] = scatter_mean(item, cluster, dim=0)
                if is_item_bool:
                    data[key] = data[key].bool()
    return data


class GridSampling3D:
    """ Clusters points into voxels with size :attr:`size`.
    Parameters
    ----------
    size: float
        Size of a voxel (in each dimension).
    quantize_coords: bool
        If True, it will convert the points into their associated sparse coordinates within the grid and store
        the value into a new `coords` attribute
    mode: string:
        The mode can be either `last` or `mean`.
        If mode is `mean`, all the points and their features within a cell will be averaged
        If mode is `last`, one random points per cell will be selected with its associated features
    """

    def __init__(self, size, quantize_coords=False, mode="mean", verbose=False):
        self._grid_size = size
        self._quantize_coords = quantize_coords
        self._mode = mode
        if verbose:
            log.warning(
                "If you need to keep track of the position of your points, use SaveOriginalPosId transform before using GridSampling3D"
            )

            if self._mode == "last":
                log.warning(
                    "The tensors within data will be shuffled each time this transform is applied. Be careful that if an attribute doesn't have the size of num_points, it won't be shuffled"
                )

    def _process(self, data):
        if self._mode == "last":
            data = shuffle_data(data)

        coords = torch.round((data.pos) / self._grid_size)
        if "batch" not in data:
            cluster = grid_cluster(coords, torch.tensor([1, 1, 1]))
        else:
            cluster = voxel_grid(coords, data.batch, 1)
        cluster, unique_pos_indices = consecutive_cluster(cluster)

        data = group_data(data, cluster, unique_pos_indices, mode=self._mode)
        if self._quantize_coords:
            data.coords = coords[unique_pos_indices].int()

        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(grid_size={}, quantize_coords={}, mode={})".format(
            self.__class__.__name__, self._grid_size, self._quantize_coords, self._mode
        )


class SaveOriginalPosId:
    """ Transform that adds the index of the point to the data object
    This allows us to track this point from the output back to the input data object
    """

    KEY = "origin_id"

    def _process(self, data):
        if hasattr(data, self.KEY):
            return data

        setattr(data, self.KEY, torch.arange(0, data.pos.shape[0]))
        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return self.__class__.__name__


class ElasticDistortion:
    """Apply elastic distortion on sparse coordinate space. First projects the position onto a 
    voxel grid and then apply the distortion to the voxel grid.

    Parameters
    ----------
    granularity: List[float]
        Granularity of the noise in meters
    magnitude:List[float]
        Noise multiplier in meters
    Returns
    -------
    data: Data
        Returns the same data object with distorted grid
    """

    def __init__(
        self, apply_distorsion: bool = True, granularity: List = [0.2, 0.4], magnitude=[0.8, 1.6],
    ):
        assert len(magnitude) == len(granularity)
        self._apply_distorsion = apply_distorsion
        self._granularity = granularity
        self._magnitude = magnitude

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        coords = coords.numpy()
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity * (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        coords = coords + interp(coords) * magnitude
        return torch.tensor(coords)

    def __call__(self, data):
        # coords = data.pos / self._spatial_resolution
        if self._apply_distorsion:
            if random.random() < 0.95:
                for i in range(len(self._granularity)):
                    data.pos = ElasticDistortion.elastic_distortion(data.pos, self._granularity[i], self._magnitude[i],)
        return data

    def __repr__(self):
        return "{}(apply_distorsion={}, granularity={}, magnitude={})".format(
            self.__class__.__name__, self._apply_distorsion, self._granularity, self._magnitude,
        )


from scipy.linalg import expm, norm


def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


class SparseVoxelizer:
    def __init__(
        self,
        voxel_size=0.05,
        clip_bound=None,
        use_augmentation=False,
        scale_augmentation_bound=(0.9, 1.1),
        rotation_augmentation_bound=((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi)),
        translation_augmentation_ratio_bound=((-0.2, 0.2), (-0.2, 0.2), (0, 0)),
        rotation_axis=2,
        ignore_label=-1,
    ):
        """
        Args:
        voxel_size: side length of a voxel
        clip_bound: boundary of the voxelizer. Points outside the bound will be deleted
            expects either None or an array like ((-100, 100), (-100, 100), (-100, 100)).
        scale_augmentation_bound: None or (0.9, 1.1)
        rotation_augmentation_bound: None or ((np.pi / 6, np.pi / 6), None, None) for 3 axis.
            Use random order of x, y, z to prevent bias.
        translation_augmentation_bound: ((-5, 5), (0, 0), (-10, 10))
        return_transformation: return the rigid transformation as well when get_item.
        ignore_label: label assigned for ignore (not a training label).
        """
        self.voxel_size = voxel_size
        self.clip_bound = clip_bound
        self.ignore_label = ignore_label
        self.rotation_axis = rotation_axis

        # Augmentation
        self.use_augmentation = use_augmentation
        self.scale_augmentation_bound = scale_augmentation_bound
        self.rotation_augmentation_bound = rotation_augmentation_bound
        self.translation_augmentation_ratio_bound = translation_augmentation_ratio_bound

    def get_transformation_matrix(self, rotation_angle=None):
        voxelization_matrix, rotation_matrix = np.eye(4), np.eye(4)
        # Get clip boundary from config or pointcloud.
        # Get inner clip bound to crop from.

        # Transform pointcloud coordinate to voxel coordinate.
        # 1. Random rotation
        rot_mat = np.eye(3)
        if self.use_augmentation and self.rotation_augmentation_bound is not None:
            if isinstance(self.rotation_augmentation_bound, collections.Iterable):
                rot_mats = []
                for axis_ind, rot_bound in enumerate(self.rotation_augmentation_bound):
                    theta = 0
                    axis = np.zeros(3)
                    axis[axis_ind] = 1
                    if rot_bound is not None:
                        theta = np.random.uniform(*rot_bound)
                    rot_mats.append(M(axis, theta))
                # Use random order
                np.random.shuffle(rot_mats)
                rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]
            else:
                raise ValueError()
        if rotation_angle is not None:
            axis = np.zeros(3)
            axis[self.rotation_axis] = 1
            rot_mat = M(axis, rotation_angle)
        rotation_matrix[:3, :3] = rot_mat
        # 2. Scale and translate to the voxel space.
        scale = 1 / self.voxel_size
        # if self.use_augmentation and self.scale_augmentation_bound is not None:
        #   scale *= np.random.uniform(*self.scale_augmentation_bound)
        np.fill_diagonal(voxelization_matrix[:3, :3], scale)
        # Since voxelization floors points, translate all points by half.
        # voxelization_matrix[:3, 3] = scale / 2
        # Get final transformation matrix.
        return voxelization_matrix, rotation_matrix

    def clip(self, coords, center=None, trans_aug_ratio=None):
        bound_min = np.min(coords, 0).astype(float)
        bound_max = np.max(coords, 0).astype(float)
        bound_size = bound_max - bound_min
        if center is None:
            center = bound_min + bound_size * 0.5
        lim = self.clip_bound
        if trans_aug_ratio is not None:
            trans = np.multiply(trans_aug_ratio, bound_size)
            center += trans
        # Clip points outside the limit
        clip_inds = [
            (coords[:, 0] >= (lim[0][0] + center[0]))
            & (coords[:, 0] < (lim[0][1] + center[0]))
            & (coords[:, 1] >= (lim[1][0] + center[1]))
            & (coords[:, 1] < (lim[1][1] + center[1]))
            & (coords[:, 2] >= (lim[2][0] + center[2]))
            & (coords[:, 2] < (lim[2][1] + center[2]))
        ]
        return clip_inds

    def voxelize(self, coords, feats, labels, center=None, rotation_angle=None, return_transformation=False):
        import MinkowskiEngine as ME

        assert coords.shape[1] == 3 and coords.shape[0] == feats.shape[0]
        if self.clip_bound is not None:
            trans_aug_ratio = np.zeros(3)
            if self.use_augmentation and self.translation_augmentation_ratio_bound is not None:
                for axis_ind, trans_ratio_bound in enumerate(self.translation_augmentation_ratio_bound):
                    trans_aug_ratio[axis_ind] = np.random.uniform(*trans_ratio_bound)

            clip_inds = self.clip(coords, center, trans_aug_ratio)
            coords, feats = coords[clip_inds], feats[clip_inds]
            if labels is not None:
                labels = labels[clip_inds]

        # Get rotation and scale
        M_v, M_r = self.get_transformation_matrix(rotation_angle=rotation_angle)
        # Apply transformations
        rigid_transformation = M_v
        if self.use_augmentation or rotation_angle is not None:
            rigid_transformation = M_r @ rigid_transformation

        homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
        coords_aug = np.floor(homo_coords @ rigid_transformation.T)[:, :3]

        coords_aug, feats, labels = ME.utils.sparse_quantize(
            coords_aug, feats, labels=labels.astype(np.int32), ignore_label=self.ignore_label
        )

        # Normal rotation
        if feats.shape[1] > 6:
            feats[:, 3:6] = feats[:, 3:6] @ (M_r[:3, :3].T)

        return_args = [coords_aug, feats, labels]
        if return_transformation:
            return_args.append(rigid_transformation.flatten())
        return tuple(return_args)

    def __call__(self, data):
        coords = data.pos.numpy()
        feats = data.x.numpy()
        labels = data.y.numpy()
        coords, x, y = self.voxelize(coords, feats, labels)
        data.coords = torch.tensor(coords)
        data.x = torch.tensor(x)
        data.y = torch.tensor(y).long()
        return data

