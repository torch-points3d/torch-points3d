import numpy as np

DEBUGGING_VARS = {"FIND_NEIGHBOUR_DIST": False}


def extract_histogram(spatial_ops, normalize=True):
    out = []
    for idx, nf in enumerate(spatial_ops["neighbour_finder"]):
        dist_meters = nf.dist_meters
        temp = {}
        for dist_meter in dist_meters:
            hist = dist_meter.histogram.copy()
            if normalize:
                hist /= hist.sum()
            temp[str(dist_meter.radius)] = hist.tolist()
            dist_meter.reset()
        out.append(temp)
    return out


class DistributionNeighbour(object):
    def __init__(self, radius, bins=1000):
        self._radius = radius
        self._bins = bins
        self._histogram = np.zeros(self._bins)

    def reset(self):
        self._histogram = np.zeros(self._bins)

    @property
    def radius(self):
        return self._radius

    @property
    def histogram(self):
        return self._histogram

    @property
    def histogram_non_zero(self):
        idx = len(self._histogram) - np.cumsum(self._histogram[::-1]).nonzero()[0][0]
        return self._histogram[:idx]

    def add_valid_neighbours(self, points):
        for num_valid in points:
            self._histogram[num_valid] += 1

    def __repr__(self):
        return "{}(radius={}, bins={})".format(self.__class__.__name__, self._radius, self._bins)
