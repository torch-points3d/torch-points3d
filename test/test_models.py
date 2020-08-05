import unittest
from omegaconf import OmegaConf
import os
import sys
from glob import glob
import torch

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

from test.mockdatasets import MockDatasetGeometric, MockDataset
from test.mockdatasets import PairMockDatasetGeometric, PairMockDataset
from test.utils import test_hasgrad

from torch_points3d.models.model_factory import instantiate_model
from torch_points3d.core.data_transform import XYZFeature, GridSampling3D
from torch_points3d.utils.model_building_utils.model_definition_resolver import resolve_model
from torch_points3d.datasets.registration.pair import Pair, PairBatch, PairMultiScaleBatch, DensePairBatch
from torch_geometric.transforms import Compose


HAS_MINKOWSKI = True
try:
    import MinkowskiEngine
except:
    HAS_MINKOWSKI = False
    print("=============== Skipping tests that require Minkowski Engine =============")

seed = 0
torch.manual_seed(seed)
device = "cpu"


def load_model_config(task, model_type, model_name):
    models_conf = os.path.join(ROOT, "conf/models/{}/{}.yaml".format(task, model_type))
    config = OmegaConf.load(models_conf)
    config.update("model_name", model_name)
    config.update("data.task", task)
    config.update("data.grid_size", 1)
    return config


def get_dataset(conv_type, task):
    num_points = 1024
    features = 2
    batch_size = 2
    if task == "object_detection":
        include_box = True
    else:
        include_box = False

    if conv_type.lower() == "dense":
        num_points = 2050
        batch_size = 1

    if task == "registration":
        if conv_type.lower() == "dense":
            return PairMockDataset(features, num_points=num_points, batch_size=batch_size)
        if conv_type.lower() == "sparse":
            tr = Compose([XYZFeature(True, True, True), GridSampling3D(size=0.01, quantize_coords=True, mode="last")])
            return PairMockDatasetGeometric(features, transform=tr, num_points=num_points, batch_size=batch_size)
        return PairMockDatasetGeometric(features, batch_size=batch_size)
    else:
        if conv_type.lower() == "dense":
            return MockDataset(
                features,
                num_points=num_points,
                include_box=include_box,
                panoptic=task == "panoptic",
                batch_size=batch_size,
            )
        if conv_type.lower() == "sparse":
            return MockDatasetGeometric(
                features,
                include_box=include_box,
                panoptic=task == "panoptic",
                transform=Compose(
                    [XYZFeature(True, True, True), GridSampling3D(size=0.01, quantize_coords=True, mode="last")]
                ),
                num_points=num_points,
                batch_size=batch_size,
            )
        return MockDatasetGeometric(
            features,
            batch_size=batch_size,
            num_points=num_points,
            include_box=include_box,
            panoptic=task == "panoptic",
        )


def has_zero_grad(model_name):
    has_zero_grad = ["PointGroup"]
    for zg in has_zero_grad:
        if zg.lower() in model_name.lower():
            return True
    return False


class TestModels(unittest.TestCase):
    def setUp(self):
        self.data_config = OmegaConf.load(os.path.join(DIR, "test_config/data_config.yaml"))
        self.model_type_files = glob(os.path.join(ROOT, "conf/models/*/*.yaml"))

    def test_runall(self):
        def is_known_to_fail(model_name):
            forward_failing = ["MinkUNet_WIP", "pointcnn", "RSConv_4LD", "RSConv_2LD", "randlanet"]
            if not HAS_MINKOWSKI:
                forward_failing += ["Res16", "MinkUNet", "ResUNetBN2B"]
            for failing in forward_failing:
                if failing.lower() in model_name.lower():
                    return True
            return False

        for type_file in self.model_type_files:
            associated_task = type_file.split("/")[-2]
            models_config = OmegaConf.load(type_file)
            models_config = OmegaConf.merge(models_config, self.data_config)
            models_config.update("data.task", associated_task)
            models_config.update("data.grid_size", 0.05)
            for model_name in models_config.models.keys():
                with self.subTest(model_name):
                    if not is_known_to_fail(model_name):
                        models_config.update("model_name", model_name)
                        dataset = get_dataset(models_config.models[model_name].conv_type, associated_task)
                        model = instantiate_model(models_config, dataset)
                        model.set_input(dataset[0], device)
                        try:
                            model.forward()
                            model.backward()
                        except Exception as e:
                            print("Forward or backward failing")
                            raise e
                        try:
                            if has_zero_grad(model_name):
                                ratio = 1
                            else:
                                ratio = test_hasgrad(model)
                            if ratio < 1:
                                print(
                                    "Model %s.%s.%s has %i%% of parameters with 0 gradient"
                                    % (associated_task, type_file.split("/")[-1][:-5], model_name, 100 * ratio)
                                )
                        except Exception as e:
                            print("Model with zero gradient %s: %s" % (type_file, model_name))
                            raise e

    def test_one_model(self):
        # Use this test to test any model when debugging
        config = load_model_config("object_detection", "votenet2", "VoteNetRSConvSmall")
        dataset = get_dataset("dense", "object_detection")
        model = instantiate_model(config, dataset)
        model.set_input(dataset[0], device)
        model.forward()
        model.backward()


if __name__ == "__main__":
    unittest.main()
