import unittest
from omegaconf import OmegaConf
import os
import sys
from glob import glob
import torch

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

from test.mockdatasets import MockDatasetGeometric

from src.models.model_factory import instantiate_model
from src.utils.model_building_utils.model_definition_resolver import resolve_model

# calls resolve_model, then find_model_using_name

seed = 0
torch.manual_seed(seed)


def load_model_config(task, model_type, model_name):
    models_conf = os.path.join(ROOT, "conf/models/{}/{}.yaml".format(task, model_type))
    config = OmegaConf.load(models_conf)
    config.update("model_name", model_name)
    config.update("data.task", "segmentation")
    return config


class TestModelUtils(unittest.TestCase):
    def setUp(self):
        self.data_config = OmegaConf.load(os.path.join(DIR, "test_config/data_config.yaml"))
        self.model_type_files = glob(os.path.join(ROOT, "conf/models/*/*.yaml"))

    def test_createall(self):
        for type_file in self.model_type_files:
            associated_task = type_file.split("/")[-2]
            models_config = OmegaConf.load(type_file)
            models_config.update("data.task", associated_task)
            models_config = OmegaConf.merge(models_config, self.data_config)
            for model_name in models_config.models.keys():
                print(model_name)
                if model_name not in ["MyTemplateModel"]:
                    models_config.update("model_name", model_name)
                    instantiate_model(models_config, MockDatasetGeometric(6))

    def test_pointnet2(self):
        params = load_model_config("segmentation", "pointnet2", "pointnet2")
        dataset = MockDatasetGeometric(5)
        model = instantiate_model(params, dataset)
        model.set_input(dataset[0])
        model.forward()
        model.backward()

    def test_kpconv(self):
        params = load_model_config("segmentation", "kpconv", "SimpleKPConv")
        dataset = MockDatasetGeometric(5)
        model = instantiate_model(params, dataset)
        model.set_input(dataset[0])
        model.forward()
        model.backward()

    def test_kpconvpretransform(self):
        params = load_model_config("segmentation", "kpconv", "SimpleKPConv")
        dataset = MockDatasetGeometric(5)
        model = instantiate_model(params, dataset)
        model.eval()
        dataset_transform = MockDatasetGeometric(5)
        dataset_transform.set_strategies(model)
        model.set_input(dataset[0])
        model.forward()
        model.get_output()

        torch.testing.assert_allclose(dataset_transform[0].pos, dataset[0].pos)
        # model.set_input(dataset_transform[0])
        # model.forward()
        # output_tr = model.get_output()
        # torch.testing.assert_allclose(output, output_tr)
        # model.backward()

    def test_largekpconv(self):
        params = load_model_config("segmentation", "kpconv", "KPConvPaper")
        params.update("data.use_category", True)
        params.update("data.first_subsampling", 0.02)
        dataset = MockDatasetGeometric(5)
        model = instantiate_model(params, dataset)
        model.set_input(dataset[0])
        model.forward()
        model.backward()

    def test_pointnet2ms(self):
        params = load_model_config("segmentation", "pointnet2", "pointnet2ms")
        dataset = MockDatasetGeometric(5)
        model = instantiate_model(params, dataset)
        model.set_input(dataset[0])
        model.forward()
        model.backward()

    def test_accumulated_gradient(self):
        params = load_model_config("segmentation", "pointnet2", "pointnet2ms")
        config_training = OmegaConf.load(os.path.join(DIR, "test_config/training_config.yaml"))
        dataset = MockDatasetGeometric(5)
        model = instantiate_model(params, dataset)
        model.instantiate_optimizers(config_training)
        model.set_input(dataset[0])
        expected_make_optimizer_step = [False, False, True, False, False, True, False, False, True, False]
        expected_contains_grads = [False, True, True, False, True, True, False, True, True, False]
        make_optimizer_steps = []
        contains_grads = []
        for epoch in range(10):
            model.forward()

            make_optimizer_step = model.manage_optimizer_zero_grad()  # Accumulate gradient if option is up
            make_optimizer_steps.append(make_optimizer_step)

            grad_ = model._modules["lin1"].weight.grad
            if grad_ is not None:
                contains_grads.append((grad_.sum() != 0).item())
            else:
                contains_grads.append(False)

            model.backward()  # calculate gradients

            if make_optimizer_step:
                model._optimizer.step()  # update parameters

        self.assertEqual(contains_grads, expected_contains_grads)
        self.assertEqual(make_optimizer_steps, expected_make_optimizer_step)


if __name__ == "__main__":
    unittest.main()
