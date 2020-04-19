from abc import ABC, abstractmethod
import math
import torch
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos, pool_batch
import torch_points_kernels as tp

from torch_points3d.utils.config import is_list
from torch_points3d.utils.enums import ConvolutionFormat


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
        from torch_geometric.nn import fps

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
