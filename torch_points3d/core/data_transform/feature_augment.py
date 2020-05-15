import random
import torch

# Those Transformation are adapted from https://github.com/chrischoy/SpatioTemporalSegmentation/blob/master/lib/transforms.py


class NormalizeRGB(object):
    """Normalize rgb between 0 and 1

    Parameters
    ----------
    normalize: bool: Whether to normalize the rgb attributes
    """

    def __init__(self, normalize=True):
        self._normalize = normalize

    def __call__(self, data):
        assert hasattr(data, "rgb")
        if not (data.rgb.max() <= 1 and data.rgb.min() >= 0):
            data.rgb = data.rgb.float() / 255.0
        return data

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self._normalize)


class ChromaticTranslation(object):
    """Add random color to the image, data must contain an rgb attribute between 0 and 1

    Parameters
    ----------
    trans_range_ratio:
        ratio of translation i.e. tramnslation = 2 * ratio * rand(-0.5, 0.5) (default: 1e-1)
    """

    def __init__(self, trans_range_ratio=1e-1):
        self.trans_range_ratio = trans_range_ratio

    def __call__(self, data):
        assert hasattr(data, "rgb")
        assert data.rgb.max() <= 1 and data.rgb.min() >= 0
        if random.random() < 0.95:
            tr = (torch.rand(1, 3) - 0.5) * 2 * self.trans_range_ratio
            data.rgb = torch.clamp(tr + data.rgb, 0, 1)
        return data

    def __repr__(self):
        return "{}(trans_range_ratio={})".format(self.__class__.__name__, self.trans_range_ratio)


class ChromaticAutoContrast(object):
    """ Rescale colors between 0 and 1 to enhance contrast

    Parameters
    ----------
    randomize_blend_factor :
        Blend factor is random
    blend_factor:
        Ratio of the original color that is kept
    """

    def __init__(self, randomize_blend_factor=True, blend_factor=0.5):
        self.randomize_blend_factor = randomize_blend_factor
        self.blend_factor = blend_factor

    def __call__(self, data):
        assert hasattr(data, "rgb")
        assert data.rgb.max() <= 1 and data.rgb.min() >= 0
        if random.random() < 0.2:
            feats = data.rgb
            lo = feats.min(0, keepdims=True)[0]
            hi = feats.max(0, keepdims=True)[0]
            assert hi.max() > 0, "invalid color value. Color is supposed to be [0-255]"

            scale = 1.0 / (hi - lo)

            contrast_feats = (feats - lo) * scale

            blend_factor = random.random() if self.randomize_blend_factor else self.blend_factor
            data.rgb = (1 - blend_factor) * feats + blend_factor * contrast_feats
        return data

    def __repr__(self):
        return "{}(randomize_blend_factor={}, blend_factor={})".format(
            self.__class__.__name__, self.randomize_blend_factor, self.blend_factor
        )


class ChromaticJitter:
    """ Jitter on the rgb attribute of data

    Parameters
    ----------
    std :
        standard deviation of the Jitter
    """

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, data):
        assert hasattr(data, "rgb")
        assert data.rgb.max() <= 1 and data.rgb.min() >= 0
        if random.random() < 0.95:
            noise = torch.randn(data.rgb.shape[0], 3)
            noise *= self.std
            data.rgb = torch.clamp(noise + data.rgb, 0, 1)
        return data

    def __repr__(self):
        return "{}(std={})".format(self.__class__.__name__, self.std)


class DropFeature:
    """ Sets the given feature to 0 with a given probability

    Parameters
    ----------
    drop_proba:
        Probability that the feature gets dropped
    feature_name:
        Name of the feature to drop
    """

    def __init__(self, drop_proba=0.2, feature_name="rgb"):
        self._drop_proba = drop_proba
        self._feature_name = feature_name

    def __call__(self, data):
        assert hasattr(data, self._feature_name)
        if random.random() < self._drop_proba:
            data[self._feature_name] = data[self._feature_name] * 0
        return data

    def __repr__(self):
        return "DropFeature: proba = {}, feature = {}".format(self._drop_proba, self._feature_name)


class Jitter:
    """
    add a small gaussian noise to the feature.
    Parameters
    ----------
    mu: float
        mean of the gaussian noise
    sigma: float
        standard deviation of the gaussian noise
    p: float
        probability of noise
    """

    def __init__(self, mu=0, sigma=0.01, p=0.95):
        self.mu = mu
        self.sigma = sigma
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            data.x += torch.randn_like(data.x) * self.sigma + self.mu
        return data

    def __repr__(self):
        return "Jitter(mu={}, sigma={})".format(self.mu, self.sigma)
