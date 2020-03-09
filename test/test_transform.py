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

from src.core.data_transform import (
    instantiate_transform,
    instantiate_transforms,
    GridSampling,
    MultiScaleTransform,
    AddFeatByKey,
)
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
            RadiusNeighbourFinder(0.5, 100, ConvolutionFormat.PARTIAL_DENSE.value),
            RadiusNeighbourFinder(0.5, 150, ConvolutionFormat.PARTIAL_DENSE.value),
            RadiusNeighbourFinder(1, 200, ConvolutionFormat.PARTIAL_DENSE.value),
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
        for i in range(len(ms[0].idx_neighboors)):
            self.assertEqual(set(ms[0].idx_neighboors[i].tolist()), set(idx[i].tolist()))
        self.assertEqual(ms[1].idx_neighboors.shape[1], 150)
        self.assertEqual(ms[2].idx_neighboors.shape[1], 200)

        upsample = transformed.upsample
        self.assertEqual(upsample[0].num_nodes, ms[1].num_nodes)
        self.assertEqual(upsample[1].num_nodes, pos.shape[0])
        self.assertEqual(upsample[1].x_idx.max(), ms[0].num_nodes - 1)
        self.assertEqual(upsample[1].y_idx.max(), pos.shape[0] - 1)
        self.assertEqual(upsample[1].__inc__("x_idx", 0), ms[0].num_nodes)
        self.assertEqual(upsample[1].__inc__("y_idx", 0), pos.shape[0])

    def test_AddFeatByKey(self):

        add_to_x = [False, True]
        feat_name = ["y", "none"]
        strict = [False, True]
        input_nc_feat = [None, 1, 2]

        c = 0
        for atx in add_to_x:
            for fn in feat_name:
                for ine in input_nc_feat:
                    for s in strict:
                        fn_none = False
                        ine_2 = False
                        try:
                            data = Data(x=torch.randn((10)), pos=torch.randn((10)), y=torch.randn((10)))
                            transform = AddFeatByKey(atx, fn, input_nc_feat=ine, strict=s)
                            data = transform(data)
                        except Exception:
                            if fn == "none":
                                fn_none = True
                            if ine == 2:
                                ine_2 = True

                        if fn_none or ine_2:
                            c += 1
                            continue

                        if not atx:
                            self.assertEqual(data.x.shape, torch.Size([10]))
                        else:
                            if fn == "none":
                                self.assertEqual(data.x.shape, torch.Size([10]))
                            else:
                                self.assertEqual(data.x.shape, torch.Size([10, 2]))

                        c += 1


if __name__ == "__main__":
    unittest.main()
