import unittest
import omegaconf
from omegaconf import OmegaConf, DictConfig
import os
import sys
import hydra
import shutil
import torch

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

from mockdatasets import MockDatasetGeometric
from torch_points3d.models.model_factory import instantiate_model
from torch_points3d.utils.model_building_utils.model_definition_resolver import resolve_model
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint


def load_config(task, model_type) -> DictConfig:
    models_conf = os.path.join(ROOT, "conf/models/{}/{}.yaml".format(task, model_type))
    if omegaconf.__version__ == '1.4.1':
        config = OmegaConf.load(models_conf)
        config.update("model_name", "pointnet2")
        config.update("data.task", "segmentation")
    else:
        config = OmegaConf.create({"models": OmegaConf.load(models_conf)})
        OmegaConf.update(config, "model_name", "pointnet2", merge=True)
        OmegaConf.update(config, "data.task", "segmentation", merge=True)

    return config


def remove(path):
    try:
        os.remove(path)
    except:
        pass


class MockModel(torch.nn.Module):
    """ Mock mdoel that does literaly nothing but holds a state
    """

    def __init__(self):
        super().__init__()
        self.state = torch.nn.parameter.Parameter(torch.tensor([1.0]))
        self.optimizer = torch.nn.Module()
        self.schedulers = {}
        self.num_epochs = None
        self.num_batches = 0
        self.num_samples = -1


class TestModelCheckpoint(unittest.TestCase):
    def setUp(self):
        self.data_config = OmegaConf.load(os.path.join(DIR, "test_config/data_config.yaml"))
        training_config = OmegaConf.load(os.path.join(DIR, "test_config/training_config.yaml"))
        scheduler_config = OmegaConf.load(os.path.join(DIR, "test_config/scheduler_config.yaml"))
        params = load_config("segmentation", "pointnet2")
        self.config = OmegaConf.merge(training_config, scheduler_config, params)
        self.model_name = "model"

    def test_model_ckpt_using_pointnet2ms(self,):
        # Create a checkpt

        self.run_path = os.path.join(DIR, "checkpt")
        if not os.path.exists(self.run_path):
            os.makedirs(self.run_path)

        model_checkpoint = ModelCheckpoint(self.run_path, self.model_name, "test", run_config=self.config, resume=False)
        dataset = MockDatasetGeometric(5)
        model = instantiate_model(self.config, dataset)
        model.set_input(dataset[0], "cpu")
        model.instantiate_optimizers(self.config)

        num_batches = 100
        num_epochs = 5
        num_samples = 3
        model.num_batches = num_batches
        model.num_epochs = num_epochs
        model.num_samples = num_samples

        mock_metrics = {"current_metrics": {"acc": 12}, "stage": "test", "epoch": 10}
        metric_func = {"acc": max}
        model_checkpoint.save_best_models_under_current_metrics(model, mock_metrics, metric_func)

        # Load checkpoint and initialize model
        model_checkpoint = ModelCheckpoint(self.run_path, self.model_name, "test", self.config, resume=True)
        model2 = model_checkpoint.create_model(dataset, weight_name="acc")

        self.assertEqual(str(model.optimizer.__class__.__name__), str(model2.optimizer.__class__.__name__))

        self.assertEqual(model2.num_batches, num_batches)
        self.assertEqual(model2.num_epochs, num_epochs)
        self.assertEqual(model2.num_samples, num_samples)
        self.assertEqual(model.optimizer.defaults, model2.optimizer.defaults)
        self.assertEqual(model.schedulers["lr_scheduler"].state_dict(), model2.schedulers["lr_scheduler"].state_dict())
        self.assertEqual(model.schedulers["bn_scheduler"].state_dict(), model2.schedulers["bn_scheduler"].state_dict())

        remove(os.path.join(ROOT, "{}.pt".format(self.model_name)))
        remove(os.path.join(DIR, "{}.pt".format(self.model_name)))

    def test_best_metric(self):
        self.run_path = os.path.join(DIR, "checkpt")
        if not os.path.exists(self.run_path):
            os.makedirs(self.run_path)

        model_checkpoint = ModelCheckpoint(self.run_path, self.model_name, "test", run_config=self.config, resume=False)
        model = MockModel()
        optimal_state = model.state.item()
        metric_func = {"acc": max}
        mock_metrics = {"current_metrics": {"acc": 12}, "stage": "test", "epoch": 10}
        model_checkpoint.save_best_models_under_current_metrics(model, mock_metrics, metric_func)
        model.state[0] = 2
        mock_metrics = {"current_metrics": {"acc": 0}, "stage": "test", "epoch": 11}
        model_checkpoint.save_best_models_under_current_metrics(model, mock_metrics, metric_func)
        mock_metrics = {"current_metrics": {"acc": 10}, "stage": "train", "epoch": 11}
        model_checkpoint.save_best_models_under_current_metrics(model, mock_metrics, metric_func)
        mock_metrics = {"current_metrics": {"acc": 15}, "stage": "train", "epoch": 11}
        model_checkpoint.save_best_models_under_current_metrics(model, mock_metrics, metric_func)
        self.assertEqual(model_checkpoint.checkpoint_path, os.path.join(self.run_path, self.model_name + ".pt"))

        ckp = torch.load(model_checkpoint.checkpoint_path)

        self.assertEqual(ckp["models"]["best_acc"]["state"].item(), optimal_state)
        self.assertEqual(ckp["models"]["latest"]["state"].item(), model.state.item())

    def test_dataset_properties(self):
        self.run_path = os.path.join(DIR, "checkpt")
        if not os.path.exists(self.run_path):
            os.makedirs(self.run_path)

        model_checkpoint = ModelCheckpoint(self.run_path, self.model_name, "test", run_config=self.config, resume=False)
        model_checkpoint.dataset_properties = {"first": 1, "num_classes": 20}
        model = MockModel()
        metric_func = {"acc": max}
        mock_metrics = {"current_metrics": {"acc": 12}, "stage": "test", "epoch": 10}
        metric_func = {"acc": max}
        model_checkpoint.save_best_models_under_current_metrics(model, mock_metrics, metric_func)

        ckp = ModelCheckpoint(self.run_path, self.model_name, "test", run_config=self.config, resume=False)

        self.assertEqual(ckp.dataset_properties, model_checkpoint.dataset_properties)

    def tearDown(self):
        if os.path.exists(self.run_path):
            shutil.rmtree(self.run_path)
            # os.remove(os.path.join(DIR, "{}.pt".format(self.model_name)))


if __name__ == "__main__":
    unittest.main()
