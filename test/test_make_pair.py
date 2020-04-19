import unittest
import torch
from torch_points3d.datasets.registration.pair import Pair


class TestMakePair(unittest.TestCase):
    def simple_test(self):
        nb_points_1 = 101
        data_source = Pair(
            pos=torch.randn((nb_points_1, 3)),
            x=torch.randn((nb_points_1, 9)),
            norm=torch.randn((nb_points_1, 3)),
            random_feat=torch.randn((nb_points_1, 15)),
        )
        nb_points_2 = 105
        data_target = Pair(
            pos=torch.randn((nb_points_2, 3)),
            x=torch.randn((nb_points_2, 9)),
            norm=torch.randn((nb_points_2, 3)),
            random_feat=torch.randn((nb_points_2, 15)),
        )

        b = Pair.make_pair(data_source, data_target)
        self.assertEqual(b.pos.size(), (nb_points_1 + nb_points_2, 3))
        self.assertEqual(b.x.size(), (nb_points_1 + nb_points_2, 9))
        print("pair:", b.pair)
        assert getattr(b, "pair", None) is not None
        self.assertEqual(b.pos_source.size(), (nb_points_1, 3))
        self.assertEqual(b.x_source.size(), (nb_points_1, 9))
        self.assertEqual(b.pos_target.size(), (nb_points_2, 3))
        self.assertEqual(b.x_target.size(), (nb_points_2, 9))


if __name__ == "__main__":
    unittest.main()
