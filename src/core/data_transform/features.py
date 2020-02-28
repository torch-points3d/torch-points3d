import torch


class OneFeature(object):
    """
        x gets ones as features
    """
    def __call__(self, data):

        ones = torch.ones(data.pos.shape[0]).unsqueeze(-1)
        if data.x is None:
            data.x = ones
        else:
            data.x = torch.cat([data.x, ones], -1)
        return data


class XYZFeature(object):
    """
    add the X, Y and Z as a feature
    """

    def __init__(self, add_x=True, add_y=True, add_z=True):
        self.axis = []
        if(add_x):
            self.axis.append(0)
        if(add_y):
            self.axis.append(1)
        if(add_z):
            self.axis.append(2)

    def __call__(self, data):
        assert data.pos is not None
        xyz = data.pos[:, self.axis]
        if data.x is None:
            data.x = xyz
        else:
            data.x = torch.cat([data.x, xyz], -1)
        return data


class RGBFeature(object):
    """
    add color as feature if it exists
    """

    def __call__(self, data):
        assert hasattr(data, 'color')
        color = data.color
        if data.x is None:
            data.x = color
        else:
            data.x = torch.cat([data.x, color], -1)
        return data


class NormalFeature(object):
    """
    add normal as feature. if it doesn't exist, compute normals
    using PCA
    """
    def __call__(self, data):
        if data.norm is None:
            raise NotImplementedError("TODO: Implement normal computation")

        norm = data.norm
        if data.x is None:
            data.x = norm
        else:
            data.x = torch.cat([data.x, norm], -1)
        return data
