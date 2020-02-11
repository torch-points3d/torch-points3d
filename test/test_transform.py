import unittest
import sys
import os
import torch_geometric.transforms as T
import numpy as np
import numpy.testing as npt
import torch
from torch_geometric.data import Data
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, ".."))

from src.core.data_transform import instantiate_transform, instantiate_transforms, GridSampling, MultiScaleTransform
from src.core.spatial_ops import RadiusNeighbourFinder, KNNInterpolate
from src.utils.enums import ConvolutionFormat
from src.datasets.multiscale_data import MultiScaleBatch


class Testhelpers(unittest.TestCase):
    def test_Instantiate(self):
        conf = DictConfig({"transform": "GridSampling", "params": {"size": 0.1}})
        t = instantiate_transform(conf)
        self.assertIsInstance(t, GridSampling)

        conf = DictConfig({"transform": "None", "params": {"size": 0.1}})
        with self.assertRaises(ValueError):
            t = instantiate_transform(conf)

    def test_InstantiateTransforms(self):
        conf = ListConfig([{"transform": "GridSampling", "params": {"size": 0.1}}, {"transform": "Center"},])
        t = instantiate_transforms(conf)
        self.assertIsInstance(t.transforms[0], GridSampling)
        self.assertIsInstance(t.transforms[1], T.Center)

    def test_multiscaleTransforms(self):
        samplers = [GridSampling(0.25), None, GridSampling(0.5)]
        search = [
            RadiusNeighbourFinder(0.5, 10, ConvolutionFormat.PARTIAL_DENSE.value[-1]),
            RadiusNeighbourFinder(0.5, 15, ConvolutionFormat.PARTIAL_DENSE.value[-1]),
            RadiusNeighbourFinder(1, 20, ConvolutionFormat.PARTIAL_DENSE.value[-1]),
        ]
        upsampler = [KNNInterpolate(1), KNNInterpolate(1)]

        N = 10
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        xv, yv = np.meshgrid(x, y)

        pos = torch.tensor([xv.flatten(), yv.flatten(), np.zeros(N * N)]).T
        x = torch.ones_like(pos)
        d = Data(pos=pos, x=x).contiguous()
        ms_transform = MultiScaleTransform({"sampler": samplers, "neighbour_finder": search, "upsample_op": upsampler})

        transformed = ms_transform(d.clone())
        npt.assert_almost_equal(transformed.x.numpy(), x.numpy())
        npt.assert_almost_equal(transformed.pos.numpy(), pos.numpy())

        ms = transformed.multiscale
        npt.assert_almost_equal(ms[0].pos.numpy(), ms[1].pos.numpy())
        npt.assert_almost_equal(ms[0].pos.numpy(), samplers[0](d.clone()).pos.numpy())
        npt.assert_almost_equal(ms[2].pos.numpy(), samplers[2](ms[0].clone()).pos.numpy())

        self.assertEqual(ms[0].__inc__("idx_neighboors", 0), pos.shape[0])
        idx, _ = search[0](
            d.pos,
            ms[0].pos,
            torch.zeros((d.pos.shape[0]), dtype=torch.long),
            torch.zeros((ms[0].pos.shape[0]), dtype=torch.long),
        )
        torch.testing.assert_allclose(ms[0].idx_neighboors, idx)
        self.assertEqual(ms[1].idx_neighboors.shape[1], 15)
        self.assertEqual(ms[2].idx_neighboors.shape[1], 20)

        upsample = transformed.upsample
        self.assertEqual(upsample[0].num_nodes, ms[1].num_nodes)
        self.assertEqual(upsample[1].num_nodes, pos.shape[0])
        self.assertEqual(upsample[1].x_idx.max(), ms[0].num_nodes - 1)
        self.assertEqual(upsample[1].y_idx.max(), pos.shape[0] - 1)
        self.assertEqual(upsample[1].__inc__("x_idx", 0), ms[0].num_nodes)
        self.assertEqual(upsample[1].__inc__("y_idx", 0), pos.shape[0])


if __name__ == "__main__":
    unittest.main()
