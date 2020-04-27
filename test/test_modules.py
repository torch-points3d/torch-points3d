import unittest
import numpy as np
import numpy.testing as npt
import torch
import os
import sys
from collections import defaultdict
from omegaconf import DictConfig

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)

from torch_points3d.models.base_model import BaseModel, BaseInternalLossModule
from torch_points3d.modules.PointNet.modules import PointNetSTN3D


class TestPointnetModules(unittest.TestCase):

    # test that stn forward works and is initialised with the identity
    def test_stn(self):
        pos = torch.tensor([[1, 1, 2], [-1, 0, 1], [10, 12, 13], [-18, 15, 16]]).to(torch.float32)
        batch = torch.tensor([0, 0, 1, 1])

        stn = PointNetSTN3D(batch_size=2)

        trans_pos = stn(pos, batch)

        npt.assert_array_equal(np.asarray(pos.detach()), np.asarray(trans_pos.detach()))


class MockLossModule(BaseInternalLossModule):
    def __init__(self, internal_losses):
        super().__init__()
        self.internal_losses = internal_losses

    def get_internal_losses(self):
        return self.internal_losses


class MockModel(BaseModel):
    def __init__(self):
        super().__init__(DictConfig({"conv_type": "dummy"}))

        self.model1 = MockLossModule({"mock_loss_1": torch.tensor(0.5), "mock_loss_2": torch.tensor(0.3),})

        self.model2 = MockLossModule({"mock_loss_3": torch.tensor(1.0),})


class TestInternalLosses(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()

    def test_get_named_internal_losses(self):

        d = defaultdict(list)
        d["mock_loss_1"].append(torch.tensor(0.5))
        d["mock_loss_2"].append(torch.tensor(0.3))
        d["mock_loss_3"].append(torch.tensor(1.0))

        lossDict = self.model.get_named_internal_losses()
        self.assertEqual(lossDict, d)

    def test_get_internal_loss(self):

        loss = self.model.get_internal_loss()
        self.assertAlmostEqual(loss.item(), 0.6)


if __name__ == "__main__":
    unittest.main()
