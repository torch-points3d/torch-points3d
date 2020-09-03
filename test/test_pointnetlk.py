import unittest


from torch_points3d.models.registration.pointnetlk import PointnetLK


class TestHelpers(unittest.TestCase):
    """
    test different components of pointnetLK
    """

    def test_approx_Jic(self):
        pass

    def test_iclk(self):
        """
        check wheter the iteration passes or not
        """


if __name__ == "__main__":
    unittest.main()
