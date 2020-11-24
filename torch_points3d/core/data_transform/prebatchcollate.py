import logging

log = logging.getLogger(__name__)


class ClampBatchSize:
    """ Drops sample in a batch if the batch gets too large

    Parameters
    ----------
    num_points : int, optional
        Maximum number of points per batch, by default 100000
    """

    def __init__(self, num_points=100000):
        self._num_points = num_points

    def __call__(self, datas):
        assert isinstance(datas, list)
        batch_id = 0
        batch_num_points = 0
        removed_sample = False
        datas_out = []
        for batch_id, d in enumerate(datas):
            num_points = datas[batch_id].pos.shape[0]
            batch_num_points += num_points
            if self._num_points and batch_num_points > self._num_points:
                batch_num_points -= num_points
                removed_sample = True
                continue
            datas_out.append(d)

        if removed_sample:
            num_full_points = sum(len(d.pos) for d in datas)
            num_full_batch_size = len(datas_out)
            log.warning(
                f"\t\tCannot fit {num_full_points} points into {self._num_points} points "
                f"limit. Truncating batch size at {num_full_batch_size} out of {len(datas)} with {batch_num_points}."
            )
        return datas_out

    def __repr__(self):
        return "{}(num_points={})".format(self.__class__.__name__, self._num_points)
