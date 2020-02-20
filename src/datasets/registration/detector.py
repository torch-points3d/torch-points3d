import torch


class RandomDetector(object):
    """
    Basic class for detector in the test set
    """

    def __init__(self, num_points=5000):
        self.num_points = num_points

    def __call__(self, data):
        keypoints_idx = torch.randperm(data.pos.size(0))[:self.num_points]
        data.keypoints = keypoints_idx
        return data
