import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)

from torch_points3d.api import create_model


class TestAPI(unittest.TestCase):
    def test_api_arguments(self):

        model = create_model(
            model_type="kpconv",
            model_architecture="unet",
            input_nc=3,
            output_nc=5,
            channel_nn=[64, 128, 256],
            pre_trained=False,
            config=None,
        )


if __name__ == "__main__":
    unittest.main()
