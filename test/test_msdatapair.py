import unittest
import torch
import torch.testing as tt
from torch_geometric.data import Data


from torch_points3d.datasets.registration.pair import Pair, MultiScalePair, PairBatch, PairMultiScaleBatch


class TestMSPair(unittest.TestCase):
    """

    """
    def test_apply(self):
        x = torch.tensor([1])
        pos = torch.tensor([1])
        x_target = torch.tensor([2])
        pos_target = torch.tensor([2])

        ms = [Data(x=2 * x, pos=2 * pos), Data(x=3 * x, pos=3 * pos)]
        ms_target = [Data(x=2 * x_target, pos=2 * pos_target),
                     Data(x=3 * x_target, pos=3 * pos_target)]
        data = MultiScalePair(x=x, pos=pos, multiscale=ms,
                              x_target=x_target, pos_target=pos_target,
                              multiscale_target=ms_target)
        data.apply(lambda x: 2 * x)
        self.assertEqual(data.x[0], 2)
        self.assertEqual(data.pos[0], 2)
        self.assertEqual(data.multiscale[0].pos[0], 4)
        self.assertEqual(data.multiscale[0].x[0], 4)
        self.assertEqual(data.multiscale[1].pos[0], 6)
        self.assertEqual(data.multiscale[1].x[0], 6)

        self.assertEqual(data.x_target[0], 4)
        self.assertEqual(data.pos_target[0], 4)
        self.assertEqual(data.multiscale_target[0].pos[0], 8)
        self.assertEqual(data.multiscale_target[0].x[0], 8)
        self.assertEqual(data.multiscale_target[1].pos[0], 12)
        self.assertEqual(data.multiscale_target[1].x[0], 12)

    def test_pair_batch(self):
        d1 = Data(x=torch.tensor([1]), pos=torch.tensor([1]))
        d2 = Data(x=torch.tensor([2]), pos=torch.tensor([4]))
        d3 = Data(x=torch.tensor([3]), pos=torch.tensor([9]))
        d4 = Data(x=torch.tensor([4]), pos=torch.tensor([16]))
        p1 = Pair.make_pair(d1, d2)
        p2 = Pair.make_pair(d3, d4)
        batch = PairBatch.from_data_list([p1, p2])
        tt.assert_allclose(batch.x, torch.tensor([1, 3]))
        tt.assert_allclose(batch.pos, torch.tensor([1, 9]))
        tt.assert_allclose(batch.batch, torch.tensor([0, 1]))
        tt.assert_allclose(batch.x_target, torch.tensor([2, 4]))
        tt.assert_allclose(batch.pos_target, torch.tensor([4, 16]))
        tt.assert_allclose(batch.batch_target, torch.tensor([0, 1]))


    def test_ms_pair_batch(self):
        x = torch.tensor([1])
        pos = x
        x_target = torch.tensor([2])
        pos_target = x
        ms = [Data(x=x, pos=pos), Data(x=4 * x, pos=4 * pos)]
        ms_target = [Data(x=x_target, pos=pos_target),
                     Data(x=4 * x_target, pos=4 * pos_target)]
        data1 = MultiScalePair(x=x, pos=pos, multiscale=ms,
                               x_target=x_target, pos_target=pos_target,
                               multiscale_target=ms_target)

        x = torch.tensor([3])
        pos = x
        x_target = torch.tensor([4])
        pos_target = x
        ms = [Data(x=x, pos=pos), Data(x=4 * x, pos=4 * pos)]
        ms_target = [Data(x=x_target, pos=pos_target),
                     Data(x=4 * x_target, pos=4 * pos_target)]
        data2 = MultiScalePair(x=x, pos=pos, multiscale=ms,
                               x_target=x_target, pos_target=pos_target,
                               multiscale_target=ms_target)

        batch = PairMultiScaleBatch.from_data_list([data1, data2])
        tt.assert_allclose(batch.x, torch.tensor([1, 3]))
        tt.assert_allclose(batch.x_target, torch.tensor([2, 4]))
        tt.assert_allclose(batch.batch, torch.tensor([0, 1]))
        tt.assert_allclose(batch.batch_target, torch.tensor([0, 1]))

        ms_batches = batch.multiscale
        tt.assert_allclose(ms_batches[0].batch, torch.tensor([0, 1]))
        tt.assert_allclose(ms_batches[1].batch, torch.tensor([0, 1]))
        tt.assert_allclose(ms_batches[1].x, torch.tensor([4, 12]))

        ms_batches = batch.multiscale_target
        tt.assert_allclose(ms_batches[0].batch, torch.tensor([0, 1]))
        tt.assert_allclose(ms_batches[1].batch, torch.tensor([0, 1]))
        tt.assert_allclose(ms_batches[1].x, torch.tensor([8, 16]))
