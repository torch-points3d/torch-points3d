import unittest
import sys
import os
from itertools import combinations
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf
import pickle
import shutil

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, ".."))
PATH_OUTPUTS = os.path.join(DIR_PATH, "data/shapenet/outputs")

from torch_points3d.trainer import Trainer


def load_pickle(p):
    infile = open(p, "rb")
    new_dict = pickle.load(infile)
    infile.close()
    return new_dict


def getcwd():
    return PATH_OUTPUTS


class TestTrainer(unittest.TestCase):
    def test_training_on_shapenet_fixed(self):

        if not os.path.exists(PATH_OUTPUTS):
            os.makedirs(PATH_OUTPUTS)

        cfg = load_pickle(os.path.join(DIR_PATH, "data/shapenet/shapenet_config.p"))
        os.getcwd = getcwd

        cfg.training.epochs = 2
        cfg.training.num_workers = 0
        cfg.training.checkpoint_dir = PATH_OUTPUTS
        cfg.data.is_test = True

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

        shutil.rmtree(PATH_OUTPUTS)


if __name__ == "__main__":
    unittest.main()
