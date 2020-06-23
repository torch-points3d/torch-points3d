import unittest
import sys
import os
from itertools import combinations
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf
import shutil

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, ".."))
from torch_points3d.trainer import Trainer


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.path = os.getcwd()

    def test_trainer_on_shapenet_fixed(self):
        self.path_outputs = os.path.join(DIR_PATH, "data/shapenet/outputs")
        if not os.path.exists(self.path_outputs):
            os.makedirs(self.path_outputs)
        os.chdir(self.path_outputs)

        cfg = OmegaConf.load(os.path.join(DIR_PATH, "data/shapenet/shapenet_config.yaml"))
        cfg.training.epochs = 2
        cfg.training.num_workers = 0
        cfg.data.is_test = True
        cfg.data.dataroot = os.path.join(DIR_PATH, "data/")

        trainer = Trainer(cfg)
        trainer.train()

        self.assertEqual(trainer.early_break, True)
        self.assertEqual(trainer.profiling, False)
        self.assertEqual(trainer.precompute_multi_scale, False)
        self.assertEqual(trainer.wandb_log, False)

        keys = [k for k in trainer._tracker.get_metrics().keys()]
        self.assertEqual(keys, ["test_loss_seg", "test_Cmiou", "test_Imiou"])
        trainer._cfg.voting_runs = 2
        trainer.eval()

    def test_trainer_on_scannet_object_detection(self):
        self.path_outputs = os.path.join(DIR_PATH, "data/scannet-fixed/outputs")
        if not os.path.exists(self.path_outputs):
            os.makedirs(self.path_outputs)
        os.chdir(self.path_outputs)

        cfg = OmegaConf.load(os.path.join(DIR_PATH, "data/scannet-fixed/config_object_detection.yaml"))
        cfg.training.epochs = 2
        cfg.training.num_workers = 0
        cfg.data.is_test = True
        cfg.data.dataroot = os.path.join(DIR_PATH, "data/")
        trainer = Trainer(cfg)
        trainer.train()

    def test_trainer_on_scannet_segmentation(self):
        self.path_outputs = os.path.join(DIR_PATH, "data/scannet/outputs")
        if not os.path.exists(self.path_outputs):
            os.makedirs(self.path_outputs)
        os.chdir(self.path_outputs)
        cfg = OmegaConf.load(os.path.join(DIR_PATH, "data/scannet/config_segmentation.yaml"))
        cfg.training.epochs = 2
        cfg.training.num_workers = 0
        cfg.data.is_test = True
        cfg.data.dataroot = os.path.join(DIR_PATH, "data/")
        trainer = Trainer(cfg)
        trainer.train()
        trainer._cfg.voting_runs = 2
        trainer._cfg.tracker_options.full_res = True
        trainer._cfg.tracker_options.make_submission = True
        trainer.eval()

    def tearDown(self):
        os.chdir(self.path)
        try:
            shutil.rmtree(self.path_outputs)
        except:
            pass


if __name__ == "__main__":
    unittest.main()
