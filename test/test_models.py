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
from torch_points3d.core.data_transform import ToSparseInput, XYZFeature, GridSampling
from torch_points3d.utils.model_building_utils.model_definition_resolver import resolve_model
from torch_points3d.datasets.registration.pair import Pair, PairBatch, PairMultiScaleBatch, DensePairBatch
from torch_geometric.transforms import Compose

# calls resolve_model, then find_model_using_name

seed = 0
torch.manual_seed(seed)
device = "cpu"


def load_model_config(task, model_type, model_name):
    models_conf = os.path.join(ROOT, "conf/models/{}/{}.yaml".format(task, model_type))
    config = OmegaConf.load(models_conf)
    config.update("model_name", model_name)
    config.update("data.task", task)
    return config


class TestModelUtils(unittest.TestCase):
    def setUp(self):
        self.data_config = OmegaConf.load(os.path.join(DIR, "test_config/data_config.yaml"))
        self.model_type_files = glob(os.path.join(ROOT, "conf/models/*/*.yaml"))

    def test_createall(self):
        for type_file in self.model_type_files:
            associated_task = type_file.split("/")[-2]
            models_config = OmegaConf.load(type_file)
            models_config = OmegaConf.merge(models_config, self.data_config)
            models_config.update("data.task", associated_task)
            for model_name in models_config.models.keys():
                with self.subTest(model_name):
                    if model_name not in ["MinkUNet_WIP"]:
                        models_config.update("model_name", model_name)
                        instantiate_model(models_config, MockDatasetGeometric(6))

    def test_runall(self):
        def is_known_to_fail(model_name):
            forward_failing = ["MinkUNet_WIP", "pointcnn", "RSConv_4LD", "RSConv_2LD", "randlanet"]
            for failing in forward_failing:
                if failing.lower() in model_name.lower():
                    return True
            return False

        def get_dataset(conv_type, task):
            features = 2
            if task == "registration":
                if conv_type.lower() == "dense":
                    return PairMockDataset(features, num_points=2048)
                if conv_type.lower() == "sparse":
                    tr = Compose(
                        [XYZFeature(True, True, True), GridSampling(size=0.01, quantize_coords=True, mode="last")]
                    )
                    return PairMockDatasetGeometric(features, transform=tr, num_points=1024)
                return PairMockDatasetGeometric(features)
            else:
                if conv_type.lower() == "dense":
                    return MockDataset(features, num_points=2048)
                if conv_type.lower() == "sparse":
                    return MockDatasetGeometric(
                        features, transform=GridSampling(size=0.01, quantize_coords=True, mode="last"), num_points=1024
                    )
                return MockDatasetGeometric(features)

        for type_file in self.model_type_files:
            associated_task = type_file.split("/")[-2]
            models_config = OmegaConf.load(type_file)
            models_config = OmegaConf.merge(models_config, self.data_config)
            models_config.update("data.task", associated_task)
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
                            ratio = test_hasgrad(model)
                            if ratio < 1:
                                print(
                                    "Model %s.%s.%s has %i%% of parameters with 0 gradient"
                                    % (associated_task, type_file.split("/")[-1][:-5], model_name, 100 * ratio)
                                )
                        except Exception as e:
                            print("Model with zero gradient %s: %s" % (type_file, model_name))
                            raise e

    def test_largekpconv(self):
        params = load_model_config("segmentation", "kpconv", "KPConvPaper")
        params.update("data.use_category", True)
        params.update("data.first_subsampling", 0.02)
        dataset = MockDatasetGeometric(5)
        model = instantiate_model(params, dataset)
        model.set_input(dataset[0], device)
        model.forward()
        model.backward()
        ratio = test_hasgrad(model)
        if ratio < 1:
            print("Model segmentation.kpconv.KPConvPaper has %i%% of parameters with 0 gradient" % (100 * ratio))

    def test_pointnet2ms(self):
        params = load_model_config("segmentation", "pointnet2", "pointnet2_largemsg")
        params.update("data.use_category", True)
        dataset = MockDataset(5, num_points=2048)
        model = instantiate_model(params, dataset)
        model.set_input(dataset[0], device)
        model.forward()
        model.backward()
        ratio = test_hasgrad(model)
        if ratio < 1:
            print(
                "Model segmentation.pointnet2.pointnet2_largemsgs has %i%% of parameters with 0 gradient"
                % (100 * ratio)
            )

    def test_siamese_minkowski(self):
        params = load_model_config("registration", "minkowski", "MinkUNet_Fragment")
        transform = Compose([XYZFeature(True, True, True), GridSampling(size=0.01, quantize_coords=True, mode="last")])
        dataset = PairMockDatasetGeometric(5, transform=transform, num_points=1024, is_pair_ind=True)
        model = instantiate_model(params, dataset)
        d = dataset[0]
        model.set_input(d, device)
        model.forward()
        model.backward()
        ratio = test_hasgrad(model)
        if ratio < 1:
            print(
                "Model registration.minkowski.MinkUNet_Fragment has %i%% of parameters with 0 gradient" % (100 * ratio)
            )


if __name__ == "__main__":
    unittest.main()
