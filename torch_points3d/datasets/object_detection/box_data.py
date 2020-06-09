import numpy as np
import torch


class BoxData:
    """ Basic data structure to hold a box prediction or ground truth
    if an objectness is provided then it will be treated as a prediction. Else, it is a ground truth box
    """

    def __init__(self, classname, corners3d, objectness=None):
        assert corners3d.shape == (8, 3)
        assert objectness is None or objectness <= 1 and objectness >= 0

        self.classname = classname
        self.corners3d = np.asarray(corners3d)

        if torch.is_tensor(objectness):
            objectness = objectness.item()
        self.objectness = objectness

    @property
    def is_gt(self):
        return self.objectness is not None

    def __repr__(self):
        return "{}: (objectness={})".format(self.__class__.__name__, self.objectness)
