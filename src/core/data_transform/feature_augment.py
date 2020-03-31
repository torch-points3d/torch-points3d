import random
import torch

# Those Transformation are adapted from https://github.com/chrischoy/SpatioTemporalSegmentation/blob/master/lib/transforms.py


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
            assert hi.max() > 0, f"invalid color value. Color is supposed to be [0-255]"

            scale = 1.0 / (hi - lo)

            contrast_feats = (feats - lo) * scale

            blend_factor = random.random() if self.randomize_blend_factor else self.blend_factor
            data.rgb = (1 - blend_factor) * feats + blend_factor * contrast_feats
        return data


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
