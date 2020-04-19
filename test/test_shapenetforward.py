import unittest
import os
import sys
import torch
from omegaconf.dictconfig import DictConfig
import numpy.testing as npt
import numpy as np
import copy

DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR, ".."))

from test.mock_models import MockModel
from torch_points3d.datasets.segmentation.forward.shapenet import _ForwardShapenet, ForwardShapenetDataset


class TestForwardData(unittest.TestCase):
    def setUp(self):
        self.datadir = os.path.join(DIR, "test_dataset")
        self.config = DictConfig(
            {
                "dataroot": self.datadir,
                "test_transforms": [{"transform": "FixedPoints", "lparams": [2]}],
                "category": ["Airplane", "Cap"],
                "forward_category": "Airplane",
            }
        )

    def test_fileList(self):
        test = _ForwardShapenet(self.datadir, 0)
        self.assertEqual(len(test), 2)

    def test_load(self):
        test = _ForwardShapenet(self.datadir, 10)
        data = test[0]
        self.assertEqual(data.sampleid, torch.tensor([0]))
        self.assertEqual(data.category[0], 10)

    def test_break(self):
        config = copy.deepcopy(self.config)
        config.forward_category = "Other"
        with self.assertRaises(ValueError):
            ForwardShapenetDataset(config)

    def test_dataloaders(self):
        dataset = ForwardShapenetDataset(self.config)
        dataset.create_dataloaders(MockModel(DictConfig({"conv_type": "DENSE"})), 2, False, 1, False)
        forward_set = dataset.test_dataloaders[0]
        for b in forward_set:
            self.assertEqual(b.origin_id.shape, (2, 2))

        sparseconfig = DictConfig({"dataroot": self.datadir, "category": "Airplane", "forward_category": "Airplane"})
        dataset = ForwardShapenetDataset(sparseconfig)
        dataset.create_dataloaders(MockModel(DictConfig({"conv_type": "PARTIAL_DENSE"})), 2, False, 1, False)
        forward_set = dataset.test_dataloaders[0]
        for b in forward_set:
            torch.testing.assert_allclose(b.origin_id, torch.tensor([0, 1, 2, 0, 1, 2, 3]))
            torch.testing.assert_allclose(b.sampleid, torch.tensor([0, 1]))

    def test_predictupsampledense(self):
        dataset = ForwardShapenetDataset(self.config)
        dataset.create_dataloaders(MockModel(DictConfig({"conv_type": "DENSE"})), 2, False, 1, False)
        forward_set = dataset.test_dataloaders[0]
        for b in forward_set:
            output = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]])
            predicted = dataset.predict_original_samples(b, "DENSE", output)
            self.assertEqual(len(predicted), 2)
            self.assertEqual(predicted["example1.txt"].shape, (3, 4))
            self.assertEqual(predicted["example2.txt"].shape, (4, 4))
            npt.assert_allclose(predicted["example1.txt"][:, -1], np.asarray([0, 0, 0]))
            npt.assert_allclose(predicted["example2.txt"][:, -1], np.asarray([1, 1, 1, 1]))

    def test_predictupsamplepartialdense(self):
        dataset = ForwardShapenetDataset(self.config)
        dataset.create_dataloaders(MockModel(DictConfig({"conv_type": "PARTIAL_DENSE"})), 2, False, 1, False)
        forward_set = dataset.test_dataloaders[0]
        for b in forward_set:
            output = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]])
            predicted = dataset.predict_original_samples(b, "PARTIAL_DENSE", output)
            self.assertEqual(len(predicted), 2)
            self.assertEqual(predicted["example1.txt"].shape, (3, 4))
            self.assertEqual(predicted["example2.txt"].shape, (4, 4))
            npt.assert_allclose(predicted["example1.txt"][:, -1], np.asarray([0, 0, 0]))
            npt.assert_allclose(predicted["example2.txt"][:, -1], np.asarray([1, 1, 1, 1]))

    def test_numclasses(self):
        dataset = ForwardShapenetDataset(self.config)
        self.assertEqual(dataset.num_classes, 8)

    def test_classtosegments(self):
        dataset = ForwardShapenetDataset(self.config)
        self.assertEqual(dataset.class_to_segments, {"Airplane": [0, 1, 2, 3], "Cap": [6, 7]})


if __name__ == "__main__":
    unittest.main()
