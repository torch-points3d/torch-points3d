

class NormalizeFeature(object):
    """Normalize a feature. By default, features will be scaled between [0,1]. Should only be applied on a dataset-level.

    Parameters
    ----------
    standardize: bool: Will use standardization rather than scaling.
    """

    def __init__(self, feature_name, standardize=False):
        self._feature_name = feature_name
        self._standardize = standardize

    def __call__(self, data):
        assert hasattr(data, self._feature_name)
        feature = data[self._feature_name]
        if self._standardize:
            feature = (feature - feature.mean()) / (feature.std())
        else:
            feature = (feature - feature.min()) / (feature.max() - feature.min())
        data[self._feature_name] = feature
        return data

    def __repr__(self):
        return "{}(feature_name={}, standardize={})".format(self.__class__.__name__, self._feature_name, self._standardize)