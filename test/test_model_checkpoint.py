import unittest
from omegaconf import OmegaConf
import os
import sys
from glob import glob
import torch
import hydra
import shutil

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

from test.mockdatasets import MockDatasetGeometric
from src import instantiate_model
from src.utils.model_building_utils.model_definition_resolver import resolve_model
from src.utils.config import set_format
from src.metrics.model_checkpoint import ModelCheckpoint

seed = 0
torch.manual_seed(seed)

def _find_model_using_name(model_class, task, model_config, dataset):
    resolve_model(model_config, dataset, task)
    return instantiate_model(model_class, task, model_config, dataset)

def load_model_config(task, model_type):
    models_conf = os.path.join(ROOT, "conf/models/{}/{}.yaml".format(task, model_type))
    config = OmegaConf.load(models_conf)
    return config.models


class TestModelCheckpoint(unittest.TestCase):
    
    def setUp(self):
        self.data_config = OmegaConf.load(os.path.join(DIR, "test_config/data_config.yaml"))
        training_config = OmegaConf.load(os.path.join(DIR, "test_config/training_config.yaml"))
        scheduler_config = OmegaConf.load(os.path.join(DIR, "test_config/scheduler_config.yaml"))

       # new_opt = OmegaConf.create({"training":training_config})
        self.config = OmegaConf.merge(training_config, scheduler_config)

    def test_model_ckpt_using_pointnet2ms(self, ):
        params = load_model_config("segmentation", "pointnet2")["pointnet2ms"]
        model_class = getattr(params, "class")
        model_config = OmegaConf.merge(params, self.data_config)
        dataset = MockDatasetGeometric(5)
        model = _find_model_using_name(model_class, "segmentation", model_config, dataset)
        model.set_input(dataset[0])
        model.instantiate_optim(self.config)
        
        # Create a checkpt
        ckpt_dir = os.path.join(DIR, "test_model_ckpt/")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        name = "test"
        model_checkpoint = ModelCheckpoint(ckpt_dir, name)
        mock_metrics = {"current_metrics":{"acc": 12}, "stage":"test", "epoch":10}
        model_checkpoint.save_best_models_under_current_metrics(model, mock_metrics)
        model_checkpoint.save_best_models_under_current_metrics(model, mock_metrics)

        # Load checkpoint and initialize model
        model_config = OmegaConf.merge(params, self.data_config)
        model2 = _find_model_using_name(model_class, "segmentation", model_config, dataset)
        model_checkpoint = ModelCheckpoint(ckpt_dir, name)
        model_checkpoint.initialize_model(model2, weight_name="acc")
        shutil.rmtree(ckpt_dir)

        assert str(model.optimizer.__class__.__name__) == str(model2.optimizer.__class__.__name__)
        assert model.optimizer.defaults == model2.optimizer.defaults
        assert model.schedulers['lr_scheduler'].state_dict() == model2.schedulers['lr_scheduler'].state_dict()

if __name__ == "__main__":
    unittest.main()
