import unittest
import torch
import torch.testing as tt
from torch_geometric.data import Data


from torch_points3d.datasets.multiscale_data import MultiScaleBatch, MultiScaleData


class TestMSData(unittest.TestCase):
    def test_apply(self):
        x = torch.tensor([1])
        pos = torch.tensor([1])
        d1 = Data(x=2 * x, pos=2 * pos)
        d2 = Data(x=3 * x, pos=3 * pos)
        data = MultiScaleData(x=x, pos=pos, multiscale=[d1, d2])
        data.apply(lambda x: 2 * x)
        self.assertEqual(data.x[0], 2)
        self.assertEqual(data.pos[0], 2)
        self.assertEqual(data.multiscale[0].pos[0], 4)
        self.assertEqual(data.multiscale[0].x[0], 4)
        self.assertEqual(data.multiscale[1].pos[0], 6)
        self.assertEqual(data.multiscale[1].x[0], 6)

    def test_batch(self):
        x = torch.tensor([1])
        pos = x
        d1 = Data(x=x, pos=pos)
        d2 = Data(x=4 * x, pos=4 * pos)
        data1 = MultiScaleData(x=x, pos=pos, multiscale=[d1, d2])

        x = torch.tensor([2])
        pos = x
        d1 = Data(x=x, pos=pos)
        d2 = Data(x=4 * x, pos=4 * pos)
        data2 = MultiScaleData(x=x, pos=pos, multiscale=[d1, d2])

        batch = MultiScaleBatch.from_data_list([data1, data2])
        tt.assert_allclose(batch.x, torch.tensor([1, 2]))
        tt.assert_allclose(batch.batch, torch.tensor([0, 1]))

        ms_batches = batch.multiscale
        tt.assert_allclose(ms_batches[0].batch, torch.tensor([0, 1]))
        tt.assert_allclose(ms_batches[1].batch, torch.tensor([0, 1]))
        tt.assert_allclose(ms_batches[1].x, torch.tensor([4, 8]))
