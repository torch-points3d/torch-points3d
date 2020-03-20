import unittest
from omegaconf import OmegaConf, DictConfig
import os
import sys
import hydra
import shutil

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

from test.mockdatasets import MockDatasetGeometric
from src.models.model_factory import instantiate_model
from src.utils.model_building_utils.model_definition_resolver import resolve_model
from src.metrics.model_checkpoint import ModelCheckpoint


def load_config(task, model_type) -> DictConfig:
    models_conf = os.path.join(ROOT, "conf/models/{}/{}.yaml".format(task, model_type))
    config = OmegaConf.load(models_conf)
    config.update("model_name", "pointnet2")
    config.update("data.task", "segmentation")
    return config


class TestModelCheckpoint(unittest.TestCase):
    def setUp(self):
        self.data_config = OmegaConf.load(os.path.join(DIR, "test_config/data_config.yaml"))
        training_config = OmegaConf.load(os.path.join(DIR, "test_config/training_config.yaml"))
        scheduler_config = OmegaConf.load(os.path.join(DIR, "test_config/scheduler_config.yaml"))
        params = load_config("segmentation", "pointnet2")
        self.config = OmegaConf.merge(training_config, scheduler_config, params)

    def test_model_ckpt_using_pointnet2ms(self,):
        # Create a checkpt
        name = "model"
        self.run_path = os.path.join(DIR, "checkpt")
        print(self.run_path)
        if not os.path.exists(self.run_path):
            os.makedirs(self.run_path)

        model_checkpoint = ModelCheckpoint(self.run_path, name, "test", run_config=self.config, resume=False)
        dataset = MockDatasetGeometric(5)
        model = instantiate_model(self.config, dataset)
        model.set_input(dataset[0], "cpu")
        model.instantiate_optimizers(self.config)

        mock_metrics = {"current_metrics": {"acc": 12}, "stage": "test", "epoch": 10}
        model_checkpoint.save_best_models_under_current_metrics(model, mock_metrics)

        # Load checkpoint and initialize model
        model_checkpoint = ModelCheckpoint(self.run_path, name, "test", self.config, resume=True)
        model2 = model_checkpoint.create_model(dataset, weight_name="acc")

        self.assertEqual(str(model.optimizer.__class__.__name__), str(model2.optimizer.__class__.__name__))
        self.assertEqual(model.optimizer.defaults, model2.optimizer.defaults)
        self.assertEqual(model.schedulers["lr_scheduler"].state_dict(), model2.schedulers["lr_scheduler"].state_dict())
        self.assertEqual(model.schedulers["bn_scheduler"].state_dict(), model2.schedulers["bn_scheduler"].state_dict())

        shutil.rmtree(self.run_path)
        os.remove(os.path.join(ROOT, "{}.pt".format(name)))


if __name__ == "__main__":
    unittest.main()
