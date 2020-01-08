from abc import ABC, abstractmethod
from typing import List, Union
import math
from functools import partial
import torch
from torch import nn
from torch_geometric.nn import fps, radius, knn
import torch_points as tp


class QueryAndGroupShadow(nn.Module):
    r"""
    Groups with a ball query of radius and shift with shadow points (0, 0, 0) and 0 features
    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, shadow_point=torch.zeros((3, )), set_zero=True, use_xyz=True):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroupShadow, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self._shadow_point = shadow_point
        self._set_zero = set_zero

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)
        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        B, C, N = features.shape
        # Return indices + 1 to allow shadow points to be added.
        idx = tp.ball_query_shifted(self.radius, self.nsample, xyz, new_xyz)

        shadow_point = self._shadow_point.unsqueeze(0).unsqueeze(0)
        shadow_point = shadow_point.repeat((xyz.shape[0], 1, 1)) .to(xyz.device)

        xyz_trans = torch.cat([shadow_point, xyz], dim=1)\
            .transpose(1, 2).contiguous()

        grouped_xyz = tp.grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self._set_zero:
            grouped_xyz[idx.unsqueeze(1).repeat((1, grouped_xyz.shape[1], 1, 1)) == 0] = 0  # Set all shadow points to 0

        if features is not None:
            shadow_features = torch.zeros((B, C, 1)).to(features.device)
            features = torch.cat([shadow_features, features], dim=-1)  # Add a shadow features at the beginning
            grouped_features = tp.grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class BaseSampler(ABC):
    """If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
    """

    def __init__(self, ratio=None, num_to_sample=None):
        if num_to_sample is not None:
            if ratio is not None:
                raise ValueError("Can only specify ratio or num_to_sample, not both")
            self._num_to_sample = num_to_sample
        else:
            self._ratio = ratio

    def __call__(self, pos, batch=None):
        return self.sample(pos, batch=batch)

    def _get_num_to_sample(self, batch_size) -> int:
        if hasattr(self, '_num_to_sample'):
            return self._num_to_sample
        else:
            return math.floor(batch_size * self._ratio)

    def _get_ratio_to_sample(self, batch_size) -> float:
        if hasattr(self, '_ratio'):
            return self._ratio
        else:
            return self._num_to_sample / float(batch_size)

    @abstractmethod
    def sample(self, pos, batch=None):
        pass


class FPSSampler(BaseSampler):
    """If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
    """

    def sample(self, pos, batch):
        if len(pos.shape) != 2:
            raise ValueError(" This class is for sparse data and expects the pos tensor to be of dimension 2")
        return fps(pos, batch, ratio=self._get_ratio_to_sample(pos.shape[0]))


class DenseFPSSampler(BaseSampler):
    """If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
    """

    def sample(self, pos, **kwargs):
        """ Sample pos

        Arguments:
            pos -- [B, N, 3]

        Returns:
            indexes -- [B, num_sample]
        """
        if len(pos.shape) != 3:
            raise ValueError(" This class is for dense data and expects the pos tensor to be of dimension 2")
        return tp.furthest_point_sample(pos, self._get_num_to_sample(pos.shape[1]))


class RandomSampler(BaseSampler):
    """If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
    """

    def sample(self, pos, batch):
        if len(pos.shape) != 2:
            raise ValueError(" This class is for sparse data and expects the pos tensor to be of dimension 2")
        idx = torch.randint(0, pos.shape[0], (self._get_num_to_sample(pos.shape[0]),))
        return idx


class DenseRandomSampler(BaseSampler):
    """If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
        Arguments:
            pos -- [B, N, 3]
    """

    def sample(self, pos, **kwargs):
        if len(pos.shape) != 3:
            raise ValueError(" This class is for dense data and expects the pos tensor to be of dimension 2")
        idx = torch.randint(0, pos.shape[1], (self._get_num_to_sample(pos.shape[1]),))
        return idx


class BaseNeighbourFinder(ABC):

    def __call__(self, x, y, batch_x, batch_y):
        return self.find_neighbours(x, y, batch_x, batch_y)

    @abstractmethod
    def find_neighbours(self, x, y, batch_x, batch_y):
        pass


class RadiusNeighbourFinder(BaseNeighbourFinder):

    def __init__(self, radius: float, max_num_neighbors: int = 64):
        self._radius = radius
        self._max_num_neighbors = max_num_neighbors

    def find_neighbours(self, x, y, batch_x, batch_y):
        return radius(x, y, self._radius, batch_x, batch_y, max_num_neighbors=self._max_num_neighbors)


class KNNNeighbourFinder(BaseNeighbourFinder):

    def __init__(self, k):
        self.k = k

    def find_neighbours(self, x, y, batch_x, batch_y):
        return knn(x, y, self.k, batch_x, batch_y)


class DilatedKNNNeighbourFinder(BaseNeighbourFinder):

    def __init__(self, k, dilation):
        self.k = k
        self.dilation = dilation
        self.initialFinder = KNNNeighbourFinder(k * dilation)

    def find_neighbours(self, x, y, batch_x, batch_y):
        # find the self.k * self.dilation closest neighbours in x for each y
        row, col = self.initialFinder.find_neighbours(x, y, batch_x, batch_y)

        # for each point in y, randomly select k of its neighbours
        index = torch.randint(self.k * self.dilation, (len(y), self.k), device=row.device, dtype=torch.long)

        arange = torch.arange(len(y), dtype=torch.long, device=row.device)
        arange = arange * (self.k * self.dilation)
        index = (index + arange.view(-1, 1)).view(-1)
        row, col = row[index], col[index]

        return row, col


class BaseMSNeighbourFinder(ABC):

    def __call__(self, x, y, batch_x=None, batch_y=None, scale_idx=0):
        return self.find_neighbours(x, y, batch_x=batch_x, batch_y=batch_y, scale_idx=scale_idx)

    @abstractmethod
    def find_neighbours(self, x, y, batch_x=None, batch_y=None, scale_idx=0):
        pass

    @property
    @abstractmethod
    def num_scales(self):
        pass


class MultiscaleRadiusNeighbourFinder(BaseMSNeighbourFinder):
    """ Radius search with support for multiscale for sparse graphs

        Arguments:
            radius {Union[float, List[float]]} 

        Keyword Arguments:
            max_num_neighbors {Union[int, List[int]]}  (default: {64})

        Raises:
            ValueError: [description]
    """

    def __init__(self, radius: Union[float, List[float]], max_num_neighbors: Union[int, List[int]] = 64):
        if not isinstance(max_num_neighbors, list) and isinstance(radius, list):
            self._radius = radius
            self._max_num_neighbors = [max_num_neighbors for i in range(len(self._radius))]
            return

        if not isinstance(radius, list) and isinstance(max_num_neighbors, list):
            self._max_num_neighbors = max_num_neighbors
            self._radius = [radius for i in range(len(self._max_num_neighbors))]
            return

        if isinstance(max_num_neighbors, list):
            if len(max_num_neighbors) != len(radius):
                raise ValueError("Both lists max_num_neighbors and radius should be of the same length")
            self._max_num_neighbors = max_num_neighbors
            self._radius = radius
            return

        self._max_num_neighbors = [max_num_neighbors]
        self._radius = [radius]

    def find_neighbours(self, x, y, batch_x=None, batch_y=None, scale_idx=0):
        if scale_idx >= self.num_scales:
            raise ValueError("Scale %i is out of bounds %i" % (scale_idx, self.num_scales))
        return radius(x, y, self._radius[scale_idx], batch_x, batch_y, max_num_neighbors=self._max_num_neighbors[scale_idx])

    @property
    def num_scales(self):
        return len(self._radius)

    def __call__(self, x, y, batch_x=None, batch_y=None, scale_idx=0):
        """ Sparse interface of the neighboorhood finder
        """
        return self.find_neighbours(x, y, batch_x, batch_y, scale_idx)


class DenseRadiusNeighbourFinder(MultiscaleRadiusNeighbourFinder):
    """ Multiscale radius search for dense graphs
    """

    def find_neighbours(self, x, y, scale_idx=0):
        if scale_idx >= self.num_scales:
            raise ValueError("Scale %i is out of bounds %i" % (scale_idx, self.num_scales))
        return tp.ball_query(self._radius[scale_idx], self._max_num_neighbors[scale_idx], x, y)

    def __call__(self, x, y, scale_idx=0, **kwargs):
        """ Dense interface of the neighboorhood finder
        """
        return self.find_neighbours(x, y, scale_idx)


class BaseMSNeighbourFinder(ABC):

    def __call__(self, x, y, batch_x=None, batch_y=None, scale_idx=0):
        return self.find_neighbours(x, y, batch_x=batch_x, batch_y=batch_y, scale_idx=scale_idx)

    @abstractmethod
    def find_neighbours(self, x, y, batch_x=None, batch_y=None, scale_idx=0):
        pass

    @property
    @abstractmethod
    def num_scales(self):
        pass


class MultiscaleRadiusNeighbourFinder(BaseMSNeighbourFinder):
    """ Radius search with support for multiscale for sparse graphs

        Arguments:
            radius {Union[float, List[float]]} 

        Keyword Arguments:
            max_num_neighbors {Union[int, List[int]]}  (default: {64})

        Raises:
            ValueError: [description]
    """

    def __init__(self, radius: Union[float, List[float]], max_num_neighbors: Union[int, List[int]] = 64):
        if not isinstance(max_num_neighbors, list) and isinstance(radius, list):
            self._radius = radius
            self._max_num_neighbors = [max_num_neighbors for i in range(len(self._radius))]
            return

        if not isinstance(radius, list) and isinstance(max_num_neighbors, list):
            self._max_num_neighbors = max_num_neighbors
            self._radius = [radius for i in range(len(self._max_num_neighbors))]
            return

        if isinstance(max_num_neighbors, list):
            if len(max_num_neighbors) != len(radius):
                raise ValueError("Both lists max_num_neighbors and radius should be of the same length")
            self._max_num_neighbors = max_num_neighbors
            self._radius = radius
            return

        self._max_num_neighbors = [max_num_neighbors]
        self._radius = [radius]

    def find_neighbours(self, x, y, batch_x=None, batch_y=None, scale_idx=0):
        if scale_idx >= self.num_scales:
            raise ValueError("Scale %i is out of bounds %i" % (scale_idx, self.num_scales))
        return radius(x, y, self._radius[scale_idx], batch_x, batch_y, max_num_neighbors=self._max_num_neighbors[scale_idx])

    @property
    def num_scales(self):
        return len(self._radius)

    def __call__(self, x, y, batch_x=None, batch_y=None, scale_idx=0):
        """ Sparse interface of the neighboorhood finder
        """
        return self.find_neighbours(x, y, batch_x, batch_y, scale_idx)


class DenseRadiusShadowNeighbourFinder(MultiscaleRadiusNeighbourFinder):
    """ Multiscale radius search for dense graphs
    """

    def __init__(self, radius: Union[float, List[float]], max_num_neighbors: Union[int, List[int]] = 64,
                 shadow_point: torch.tensor = torch.zeros((3, )), set_zero: bool = True):
        super(DenseRadiusShadowNeighbourFinder, self).__init__(radius=radius, max_num_neighbors=max_num_neighbors)

        self.groupers = []
        for radius, nsamples in zip(self._radius, self._max_num_neighbors):
            self.groupers.append(QueryAndGroupShadow(radius, nsamples, shadow_point=shadow_point, set_zero=True))

    def find_neighbours(self, xyz, new_xyz, features, scale_idx=0):
        if scale_idx >= self.num_scales:
            raise ValueError("Scale %i is out of bounds %i" % (scale_idx, self.num_scales))
        return self.groupers[scale_idx](xyz, new_xyz, features)

    def __call__(self, xyz, new_xyz, features, scale_idx=0, **kwargs):
        """ Dense interface of the neighboorhood finder
        """
        return self.find_neighbours(xyz, new_xyz, features, scale_idx)
