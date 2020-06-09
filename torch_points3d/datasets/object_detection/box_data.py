class BoxData:
    """ Basic data structure to hold a box prediction or ground truth
    if a score is provided then it will be treated as a prediction. Else, it is a ground truth box
    """

    def __init__(self, classname, corners3d, score=None):
        assert corners3d.shape == (8, 3)
        assert score <= 1 and score >= 0

        self.classname = classname
        self.corners3d = corners3d
        self.score = score

    @property
    def is_gt(self):
        return self.score is not None
