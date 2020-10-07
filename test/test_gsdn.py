import unittest
import torch
import MinkowskiEngine as ME
import math

from torch_points3d.modules.GSDN.gsdn_results import GSDNLayerPrediction, GSDNResults


class TestGSDN(unittest.TestCase):
    def test_fromlogits(self):
        coords = torch.tensor([[0, 1, 1, 0]]).int()
        feats = torch.ones((coords.shape[0], 1))
        sparsetensor = ME.SparseTensor(feats=feats, coords=coords)
        logits = torch.tensor([[0, 1, 0, 1, 1, 1, 1, 0, 1]]).float()
        sparsity = torch.tensor([[1]]).float()

        result = GSDNLayerPrediction.create_from_logits(sparsetensor, logits, 1, sparsity)
        result.grid_size = 0.1
        result.anchors = torch.tensor([[1.0, 2.0, 3.0]])

        centres, sizes = result._get_for_sample(0)
        torch.testing.assert_allclose(centres, torch.tensor([[0.1, 2.1, 0]]))
        torch.testing.assert_allclose(sizes, math.exp(1) * torch.tensor([[1, 2, 3]]))

        logits = torch.tensor([[0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]]).float()
        result = GSDNLayerPrediction.create_from_logits(sparsetensor, logits, 2, sparsity)
        result.grid_size = 0.1
        result.anchors = torch.tensor([[1.0, 2.0, 3.0], [10.0, 1, 1]])
        centres, sizes = result._get_for_sample(0)
        torch.testing.assert_allclose(centres, torch.tensor([[0.1, 2.1, 0], [10.1, 0.1, 0]]))
        torch.testing.assert_allclose(
            sizes, torch.tensor([[1, 2, 3], [math.exp(1) * 10, math.exp(1) * 1, math.exp(1) * 1]])
        )
        torch.testing.assert_allclose(result.objectness, torch.tensor([1, 0]).float())
        torch.testing.assert_allclose(result.class_logits, torch.tensor([[0, 1,], [1, 0]]).float())

    def test_evaluatelabels(self):
        coords = torch.tensor([[0, 1, 1, 0], [1, 10, 0, 0]]).int()
        feats = torch.ones((coords.shape[0], 1))
        sparsetensor = ME.SparseTensor(feats=feats, coords=coords)
        sparsity = torch.tensor([[1, 1]]).float()
        logits = torch.tensor(
            [
                [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
            ]
        ).float()
        result = GSDNLayerPrediction.create_from_logits(sparsetensor, logits, 2, sparsity)
        result.grid_size = 0.1
        result.anchors = torch.tensor([[1.0, 2.0, 3.0], [10.0, 1, 1]])

        centre_labels = [torch.tensor([[0.2, 2, 0], [0.1, 3, 10]]), torch.tensor([[10.0, 0, 0]])]
        size_labels = [torch.tensor([[1, 2, 3], [1.0, 3.0, 1.0]]), torch.tensor([[10, 1, 1]]).float()]
        class_labels = [torch.tensor([1, 0]), torch.tensor([0])]

        result.evaluate_labels(centre_labels, size_labels, class_labels)
        torch.testing.assert_allclose(result.positive_mask, torch.tensor([True, False, False, True]))
        torch.testing.assert_allclose(result.negative_mask, torch.tensor([False, True, True, False]))
        torch.testing.assert_allclose(result.sparsity_negative, torch.tensor([False, False]))
        torch.testing.assert_allclose(result.sparsity_positive, torch.tensor([True, True]))
        torch.testing.assert_allclose(result.class_labels, torch.tensor([1, 1, 0, 0]))
        torch.testing.assert_allclose(
            result.centre_labels, torch.tensor([[0.2, 2, 0], [0.2, 2, 0], [10, 0, 0], [10, 0, 0]])
        )
        torch.testing.assert_allclose(
            result.size_labels, torch.tensor([[1, 2, 3], [1, 2, 3], [10, 1, 1], [10, 1, 1]]).float()
        )
        torch.testing.assert_allclose(result.get_rescaled_size_labels()[0], torch.tensor([0, 0, 0]).float())
        torch.testing.assert_allclose(result.get_rescaled_centre_labels()[0], torch.tensor([0.1, 1.9 / 2.0, 0]).float())

    def test_losses(self):
        coords = torch.tensor([[0, 1, 1, 0], [1, 10, 0, 0]]).int()
        feats = torch.ones((coords.shape[0], 1))
        sparsetensor = ME.SparseTensor(feats=feats, coords=coords)
        sparsity = torch.tensor([[1, 1]]).float()
        logits = torch.tensor(
            [
                [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1000, -1000, -100, 100, 100, -100],
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1000, 1000, -100, 100, 100, -100],
            ]
        ).float()
        result = GSDNLayerPrediction.create_from_logits(sparsetensor, logits, 2, sparsity)
        result.grid_size = 0.1
        result.anchors = torch.tensor([[1.0, 2.0, 3.0], [10.0, 1, 1]])

        centre_labels = [torch.tensor([[0.1, 2.1, 0], [0.1, 3, 10]]), torch.tensor([[11, 0, 0]])]
        size_labels = [torch.tensor([[1, 2, 3], [1.0, 3.0, 1.0]]), torch.tensor([[10, 1, 1]]).float()]
        class_labels = [torch.tensor([1, 0]), torch.tensor([0])]

        results = GSDNResults([result])
        results.evaluate_labels(centre_labels, size_labels, class_labels, sparsetensor.coords_man)

        self.assertEqual(results.get_anchor_loss().item(), 0)
        self.assertEqual(results.get_sparsity_loss().item(), 0)
        self.assertEqual(results.get_semantic_loss().item(), 0)
        self.assertAlmostEqual(results.get_regression_loss().item(), 0)

    def test_setsparsity(self):
        anchors = torch.tensor([[1.0, 2.0, 3.0], [10.0, 1, 1]])
        nb_anchors = len(anchors)
        nb_classes = 2
        coords = torch.tensor([[0, 0, 0, 0]]).int()
        feats = torch.ones((coords.shape[0], 1))
        sparsetensor = ME.SparseTensor(feats=feats, coords=coords, tensor_stride=2)
        convtr = ME.MinkowskiConvolutionTranspose(1, 1, kernel_size=3, stride=2, generate_new_coords=True, dimension=3)
        child = convtr(sparsetensor)

        logits = torch.rand((coords.shape[0], (7 + nb_classes) * nb_anchors))
        sparsity = torch.tensor([[1]]).float()
        result = GSDNLayerPrediction.create_from_logits(sparsetensor, logits, 2, sparsity)
        result.grid_size = 0.1
        result.anchors = anchors
        result.sparsity_positive = torch.zeros(coords.shape[0]).bool()
        result.sparsity_negative = torch.ones(coords.shape[0]).bool()

        child_logits = torch.rand((child.C.shape[0], (7 + nb_classes) * nb_anchors))
        sparsity = torch.rand((child_logits.shape[0]))
        child_result = GSDNLayerPrediction.create_from_logits(child, child_logits, 2, sparsity)
        child_result.grid_size = 0.1
        child_result.anchors = anchors
        child_result.sparsity_positive = torch.rand(child.C.shape[0]) > 0.5
        mask_0 = torch.abs(child.C).sum(-1) == 0
        child_result.sparsity_positive[mask_0] = True

        result.set_sparsity(child_result, sparsetensor.coords_man)
        self.assertEqual(result.sparsity_negative.item(), False)
        self.assertEqual(result.sparsity_positive.item(), True)
