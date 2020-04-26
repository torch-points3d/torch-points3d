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

from torch_points3d.core.data_transform import (
    instantiate_transform,
    instantiate_transforms,
    GridSampling,
    MultiScaleTransform,
    Random3AxisRotation,
    AddFeatByKey,
    AddFeatsByKeys,
    RemoveAttributes,
    RandomDropout,
    ShiftVoxels,
    PCACompute,
    RandomCoordsFlip,
    RemoveDuplicateCoords,
    XYZFeature,
    ScalePos,
)
from torch_points3d.core.spatial_ops import RadiusNeighbourFinder, KNNInterpolate
from torch_points3d.utils.enums import ConvolutionFormat
from torch_points3d.datasets.multiscale_data import MultiScaleBatch

np.random.seed(0)


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
        idx = search[0](
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

    def test_AddFeatsByKeys(self):
        N = 10
        mapping = {"a": 1, "b": 2, "c": 3, "d": 4}
        keys, values = np.asarray(list(mapping.keys())), np.asarray(list(mapping.values()))
        data = Data(
            a=torch.randn((N, 1)),
            b=torch.randn((N, 2)),
            c=torch.randn((N, 3)),
            d=torch.randn((N, 4)),
            pos=torch.randn((N)),
        )
        mask = np.random.uniform(0, 1, (4)) > 0.1
        transform = AddFeatsByKeys(mask, keys)
        data_out = transform(data)
        self.assertEqual(data_out.x.shape[-1], np.sum(values[mask]))

    def test_RemoveAttributes(self):
        N = 10
        mapping = {"a": 1, "b": 2, "c": 3, "d": 4}
        keys, values = np.asarray(list(mapping.keys())), np.asarray(list(mapping.values()))
        data = Data(
            a=torch.randn((N, 1)),
            b=torch.randn((N, 2)),
            c=torch.randn((N, 3)),
            d=torch.randn((N, 4)),
            pos=torch.randn((N)),
        )
        mask = np.random.uniform(0, 1, (4)) > 0.5
        transform = RemoveAttributes(keys[mask])
        data_out = transform(data)
        for key in keys[mask]:
            self.assertNotIn(key, list(data_out.keys))

    def test_RemoveDuplicateCoords(self):
        indices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 0, 0]])
        data = Data(pos=torch.from_numpy(indices))
        transform = RemoveDuplicateCoords()
        data_out = transform(data)
        self.assertEqual(data_out.pos.shape[0], 5)

    def test_dropout(self):
        indices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 0, 0]])
        data = Data(pos=torch.from_numpy(indices))
        tr = RandomDropout(dropout_ratio=0.5, dropout_application_ratio=1.1)
        data = tr(data)
        self.assertEqual(len(data.pos), 3)

    def test_shiftvoxels(self):
        indices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 0, 0]])
        data = Data(pos=torch.from_numpy(indices).int())
        tr = ShiftVoxels()
        tr_data = tr(data.clone())
        self.assertGreaterEqual(tr_data.pos[0][0], data.pos[0][0])

    def test_PCACompute(self):
        vec1 = torch.randn(3)
        vec1 = vec1 / torch.norm(vec1)
        vec2 = torch.randn(3)
        vec2 = vec2 / torch.norm(vec2)
        norm = vec1.cross(vec2) / torch.norm(vec1.cross(vec2))
        plane = torch.randn(100, 1) * vec1.view(1, 3) + torch.randn(100, 1) * vec2.view(1, 3)
        data = Data(pos=plane)
        pca = PCACompute()
        data = pca(data)
        npt.assert_almost_equal(np.abs(data.eigenvectors[:, 0].dot(norm).item()), 1)

    def test_Random3AxisRotation(self):

        pos = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float)
        data = Data(pos=torch.from_numpy(pos).float())
        t = Random3AxisRotation(apply_rotation=True, rot_x=0, rot_y=0, rot_z=180)

        u, v, w = data.pos
        u2, v2, w2 = t(data.clone()).pos

        self.assertEqual(np.array_equal(u, u2), False)
        self.assertEqual(np.array_equal(v, v2), False)
        npt.assert_array_equal(w, w2)

        t = Random3AxisRotation(apply_rotation=True, rot_x=180, rot_y=180, rot_z=180)

        u2, v2, w2 = t(data.clone()).pos

        self.assertEqual(np.array_equal(u, u2), False)
        self.assertEqual(np.array_equal(v, v2), False)
        self.assertEqual(np.array_equal(w, w2), False)

        with self.assertRaises(Exception):
            t = Random3AxisRotation(apply_rotation=True, rot_x=None, rot_y=None, rot_z=None)

        pos = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]).astype(np.float)
        data = Data(pos=torch.from_numpy(pos).float())
        t = Random3AxisRotation(apply_rotation=True, rot_x=0, rot_y=0, rot_z=180)

        self.assertEqual(t(data.clone()).pos.shape, torch.Size([4, 3]))

    def test_RandomCoordsFlip(self):

        pos = torch.from_numpy(np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        pos_target = torch.from_numpy(np.asarray([[6, 2, 3], [3, 5, 6], [0, 8, 9]]))
        data = Data(pos=pos)

        upright_axis = ["y", "z"]
        t = RandomCoordsFlip(upright_axis, p=1)

        pos_out = t(data.clone()).pos

        self.assertEqual(np.array_equal(pos_out, pos_target), True)

    def test_XYZFeature(self):

        pos = torch.from_numpy(np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

        data = Data(pos=pos)
        t = XYZFeature()

        data_out = t(data.clone())

        x = data_out.z

        self.assertEqual(np.array_equal(x, pos[:, -1]), True)

        pos += 1

        self.assertEqual(np.array_equal(x, pos[:, -1]), False)

        self.assertIn("z", data_out.keys)
        self.assertIn("pos", data_out.keys)
        self.assertNotIn("x", data_out.keys)
        self.assertNotIn("y", data_out.keys)

    def test_scalePos(self):
        tr = ScalePos(scale=2.0)
        d = Data(pos=torch.tensor([[1, 0, 0], [0, 1, 1]]).float())
        d = tr(d)
        torch.testing.assert_allclose(d.pos, torch.tensor([[2, 0, 0], [0, 2, 2]]).float())


if __name__ == "__main__":
    unittest.main()
