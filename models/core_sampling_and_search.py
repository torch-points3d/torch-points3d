from abc import ABC, abstractmethod
from typing import List, Union
import math
from functools import partial
import torch
from torch_geometric.nn import fps, radius, knn, voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos, pool_batch
import torch_points as tp
from omegaconf import ListConfig

from utils_folder.enums import ConvolutionFormat


class BaseSampler(ABC):
    """If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
    """

    def __init__(self, ratio=None, num_to_sample=None, subsampling_param=None):
        if num_to_sample is not None:
            if (ratio is not None) or (subsampling_param is not None):
                raise ValueError("Can only specify ratio or num_to_sample or subsampling_param, not several !")
            self._num_to_sample = num_to_sample

        elif ratio is not None:
            self._ratio = ratio

        elif subsampling_param is not None:
            self._subsampling_param = subsampling_param

        else:
            raise Exception('At least ["ratio, num_to_sample, subsampling_param"] should be defined')

    def __call__(self, pos, x=None, batch=None):
        return self.sample(pos, batch=batch, x=x)

    def _get_num_to_sample(self, batch_size) -> int:
        if hasattr(self, "_num_to_sample"):
            return self._num_to_sample
        else:
            return math.floor(batch_size * self._ratio)

    def _get_ratio_to_sample(self, batch_size) -> float:
        if hasattr(self, "_ratio"):
            return self._ratio
        else:
            return self._num_to_sample / float(batch_size)

    @abstractmethod
    def sample(self, pos, x=None, batch=None):
        pass


class FPSSampler(BaseSampler):
    """If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
    """

    def sample(self, pos, batch, **kwargs):
        if len(pos.shape) != 2:
            raise ValueError(" This class is for sparse data and expects the pos tensor to be of dimension 2")
        return fps(pos, batch, ratio=self._get_ratio_to_sample(pos.shape[0]))


class GridSampler(BaseSampler):
    """If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
    """

    def sample(self, pos=None, x=None, batch=None):
        if len(pos.shape) != 2:
            raise ValueError("This class is for sparse data and expects the pos tensor to be of dimension 2")

        pool = voxel_grid(pos, batch, self._subsampling_param)
        pool, perm = consecutive_cluster(pool)
        batch = pool_batch(perm, batch)
        if x is not None:
            return pool_pos(pool, x), pool_pos(pool, pos), batch
        else:
            return None, pool_pos(pool, pos), batch


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

    def sample(self, pos, batch, **kwargs):
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
    def __init__(
        self, radius: float, max_num_neighbors: int = 64, conv_type=ConvolutionFormat.MESSAGE_PASSING.value[-1]
    ):
        self._radius = radius
        self._max_num_neighbors = max_num_neighbors
        self._conv_type = conv_type.lower()

    def find_neighbours(self, x, y, batch_x=None, batch_y=None):
        if self._conv_type == ConvolutionFormat.MESSAGE_PASSING.value[-1]:
            return radius(x, y, self._radius, batch_x, batch_y, max_num_neighbors=self._max_num_neighbors)
        elif self._conv_type == ConvolutionFormat.DENSE.value[-1] or ConvolutionFormat.PARTIAL_DENSE.value[-1]:
            return tp.ball_query(
                self._radius, self._max_num_neighbors, x, y, mode=self._conv_type, batch_x=batch_x, batch_y=batch_y
            )
        else:
            raise NotImplementedError


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
        index = torch.randint(self.k * self.dilation, (len(y), self.k), device=row.device, dtype=torch.long,)

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


def is_list(entity):
    return isinstance(entity, list) or isinstance(entity, ListConfig)


class MultiscaleRadiusNeighbourFinder(BaseMSNeighbourFinder):
    """ Radius search with support for multiscale for sparse graphs

        Arguments:
            radius {Union[float, List[float]]}

        Keyword Arguments:
            max_num_neighbors {Union[int, List[int]]}  (default: {64})

        Raises:
            ValueError: [description]
    """

    def __init__(
        self, radius: Union[float, List[float]], max_num_neighbors: Union[int, List[int]] = 64,
    ):
        if not is_list(max_num_neighbors) and is_list(radius):
            self._radius = radius
            self._max_num_neighbors = [max_num_neighbors for i in range(len(self._radius))]
            return

        if not is_list(radius) and is_list(max_num_neighbors):
            self._max_num_neighbors = max_num_neighbors
            self._radius = [radius for i in range(len(self._max_num_neighbors))]
            return

        if is_list(max_num_neighbors):
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
        return radius(
            x, y, self._radius[scale_idx], batch_x, batch_y, max_num_neighbors=self._max_num_neighbors[scale_idx],
        )

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
        num_neighbours = self._max_num_neighbors[scale_idx]
        neighbours = tp.ball_query(self._radius[scale_idx], num_neighbours, x, y)
        # mean_count = (
        #     (neighbours[0, :, :] != neighbours[0, :, 0].view((-1, 1)).repeat(1, num_neighbours)).sum(1).float().mean()
        # )
        # for i in range(1, neighbours.shape[0]):
        #     start = neighbours[i, :, 0]
        #     valid_neighbours = (neighbours[i, :, :] != start.view((-1, 1)).repeat(1, num_neighbours)).sum(1)
        #     mean_count += valid_neighbours.float().mean()
        # mean_count = mean_count / neighbours.shape[0]
        # print(
        #     "Radius: %f, Num_neighbours %i, actual, %f" % (self._radius[scale_idx], num_neighbours, mean_count.item())
        # )
        return neighbours

    def __call__(self, x, y, scale_idx=0, **kwargs):
        """ Dense interface of the neighboorhood finder
        """
        return self.find_neighbours(x, y, scale_idx)
