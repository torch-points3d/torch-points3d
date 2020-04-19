import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)

from torch_points3d.applications.kpconv import KPConv


class TestAPI(unittest.TestCase):
    def test_api_arguments(self):

        model = KPConv(
            architecture="unet",
            input_nc=3,
            output_nc=5,
            num_layers=4,
            use_rgb=True,
            use_normal=True,
            use_z=True,
            weights=False,
            config=None,
        )


if __name__ == "__main__":
    unittest.main()
