import torch


class RandomDetector(object):
    """
    Random selector for test points
    """

    def __init__(self, num_points=5000):
        self.num_points = num_points

    def __call__(self, data):
        keypoints_idx = torch.randint(0,
                                      data.pos.shape[0],
                                      (self.num_points, ))
        data.keypoints = keypoints_idx
        return data
