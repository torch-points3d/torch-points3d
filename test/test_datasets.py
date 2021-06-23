import unittest
import hydra
import sys
import os

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

from torch_points3d.datasets.panoptic.scannet import ScannetDataset
from torch_points3d.models.panoptic.structures import PanopticLabels


# class TestScannetPanoptic(unittest.TestCase):
#     def test_dataset(self):
#         with hydra.initialize(config_path="../conf"):
#             data_config = hydra.compose(config_name="data/panoptic/scannet-sparse.yaml", overrides=["data.dataroot=test/data", "+is_test=True"])

#             dataset = ScannetDataset(data_config.data)
#             self.assertEqual(len(dataset.train_dataset), 2)
#             self.assertEqual(len(dataset.val_dataset), 2)

#             data = dataset.train_dataset[0]
#             for key in PanopticLabels._fields:
#                 self.assertIn(key, data)


if __name__ == "__main__":
    unittest.main()
