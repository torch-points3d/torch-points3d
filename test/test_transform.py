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
from src.core.neighbourfinder import RadiusNeighbourFinder
from src.utils.enums import ConvolutionFormat


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
        samplers = [GridSampling(0.2), None, GridSampling(0.5)]
        search = [
            RadiusNeighbourFinder(0.1, 10, ConvolutionFormat.PARTIAL_DENSE.value[-1]),
            RadiusNeighbourFinder(0.1, 15, ConvolutionFormat.PARTIAL_DENSE.value[-1]),
            RadiusNeighbourFinder(0.1, 20, ConvolutionFormat.PARTIAL_DENSE.value[-1]),
        ]

        N = 10
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        xv, yv = np.meshgrid(x, y)

        pos = torch.tensor([xv.flatten(), yv.flatten(), np.zeros(N * N)]).T
        x = torch.ones_like(pos)
        batch = torch.tensor(
            [0 for i in range(pos.shape[0] // 2)] + [1 for i in range(pos.shape[0] // 2, pos.shape[0])]
        )
        d = Data(pos=pos, x=x, batch=batch)
        ms_transform = MultiScaleTransform({"sampler": samplers, "neighbour_finder": search})

        transformed = ms_transform(d)
        npt.assert_almost_equal(transformed.x.numpy(), x.numpy())
        npt.assert_almost_equal(transformed.pos.numpy(), pos.numpy())
        npt.assert_almost_equal(transformed.batch.numpy(), batch.numpy())

        ms = transformed.multiscale
        npt.assert_almost_equal(ms[0].pos.numpy(), ms[1].pos.numpy())
        self.assertEqual(ms[0].idx_neighboors.shape[1], 10)
        self.assertEqual(ms[1].idx_neighboors.shape[1], 15)
        self.assertEqual(ms[2].idx_neighboors.shape[1], 20)


if __name__ == "__main__":
    unittest.main()
