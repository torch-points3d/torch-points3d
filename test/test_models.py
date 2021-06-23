import unittest
import omegaconf
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
from test.utils import load_hydra_config

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
        self.model_type_files = glob(os.path.join(ROOT, "conf/model/*/*/*.yaml"))
        self.model_type_files = [tuple(x.split(os.path.sep)[-3:]) for x in self.model_type_files]

    def test_runall(self):
        def is_known_to_fail(model_name):
            forward_failing = [
                "path_pretrained",
                "MinkUNet_WIP",
                "pointcnn",
                "RSConv_4LD",
                "RSConv_2LD",
                "randlanet",
                "PVCNN",
                "ResUNet32",
                "VoteNet2Paper",
                "default",
            ]
            if not HAS_MINKOWSKI:
                forward_failing += ["Res16", "MinkUNet", "ResUNetBN2B", "ResUNet32", "Res16UNet34"]
                for cm in [2, 4, 6]:
                    for h in [1, 2, 3, 4]:
                        for s in ["", "_unshared"]:
                            forward_failing += ["MS_SVCONV_B{}cm_X2_{}head{}".format(cm, h, s)]
            for failing in forward_failing:
                if failing.lower() in model_name.lower():
                    return True
            return False

        def is_torch_sparse_backend(model_name):
            torchsparse_backend = [
                "ResUNet32",
                "Res16UNet34",
            ]
            for cm in [2, 4, 6]:
                for h in [1, 2, 3, 4]:
                    for s in ["", "_unshared"]:
                        torchsparse_backend += ["MS_SVCONV_B{}cm_X2_{}head{}".format(cm, h, s)]
            for backend in torchsparse_backend:
                if backend.lower() in model_name.lower():
                    return True
            return False

        for task, model_type, model_name in self.model_type_files:
            models_config = load_hydra_config("model", task, model_type, model_name, overrides={"+data.task=" + task, "+data.first_subsampling=0.05", "+data.use_category=False"})

            models = models_config.get("model")
            models_keys = models.keys() if models is not None else []

            if model_name == 'defaults':
                # workaround for recursive defaults
                continue

            with self.subTest(model_name):
                if not is_known_to_fail(model_name):
                    OmegaConf.update(models_config, "model_name", model_name, merge=True)
                    # modify the backend in minkowski to have the forward
                    if is_torch_sparse_backend(model_name):
                        models_config.model.backend = "minkowski"
                    dataset = get_dataset(models_config.model.conv_type, task)
                    try:
                        model = instantiate_model(models_config, dataset)
                    except Exception as e:
                        print(e)
                        raise Exception(models_config)
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
                                % (task, model_type, model_name, 100 * ratio)
                            )
                    except Exception as e:
                        print("Model with zero gradient %s: %s" % (type_file, model_name))
                        raise e

    def test_one_model(self):
        # Use this test to test any model when debugging
        config = load_hydra_config("model", "segmentation", "pointnet2", "pointnet2_charlesssg", overrides=["+model_name=pointnet2_charlesssg", 
                                    "+data.task=segmentation", "+data.first_subsampling=1", "+data.use_category=False"])

        dataset = get_dataset("dense", "segmentation")
        model = instantiate_model(config, dataset)
        model.set_input(dataset[0], device)
        model.forward()
        model.backward()


if __name__ == "__main__":
    unittest.main()
