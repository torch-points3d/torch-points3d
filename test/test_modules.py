import unittest

import torch

import os
import sys
ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(ROOT)

from models.core_modules import BaseInternalLossModule
from models.base_model import BaseModel

class MockLossModule(torch.nn.Module, BaseInternalLossModule):

    def __init__(self, internal_losses):
        super().__init__()
        self.internal_losses = internal_losses

    def get_internal_losses(self):
        return self.internal_losses

class MockModel(BaseModel):

    def __init__(self):
        super().__init__({})

        self.model1 = MockLossModule({
            'mock_loss_1': torch.tensor(0.5),
            'mock_loss_2': torch.tensor(0.3),
        })

        self.model2 = MockLossModule({
            'mock_loss_3': torch.tensor(1.0),
        })

class TestInternalLosses(unittest.TestCase):

    def setUp(self):
        self.model = MockModel()

    def test_get_named_internal_losses(self):

        lossDict = self.model.get_named_internal_losses()
        self.assertEqual(lossDict, {'mock_loss_3': 1, 'mock_loss_1': 0.5, 'mock_loss_2': 0.3})

    def test_get_internal_losses(self):

        loss = self.model.get_internal_loss()
        self.assertAlmostEqual(loss.item(), 0.6)

        
if __name__ == '__main__':
    unittest.main()




