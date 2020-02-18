import unittest
import os
import shutil
import sys
from omegaconf import OmegaConf
import torch
from torch_geometric.data import Data

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

from src.viz import Visualizer

batch_size = 2
epochs = 5
num_points = 20


def run(iter, visualizer, epoch, stage, data):
    visualizer.reset(epoch, stage)
    for i in range(iter):
        visualizer.save_visuals(data)


class TestVisualizer(unittest.TestCase):
    def test_empty(self):
        mock_data = Data()
        mock_data.pos = torch.zeros((batch_size, num_points, 3))
        mock_data.y = torch.zeros((batch_size, num_points, 1))
        mock_data.pred = torch.zeros((batch_size, num_points, 1))
        data = {}

        run_path = os.path.join(DIR, "test_viz")

        try:
            shutil.rmtree(run_path)
        except:
            pass
        if not os.path.exists(run_path):
            os.makedirs(run_path)

        mock_num_batches = {"train": 9, "test": 3, "val": 0}
        config = OmegaConf.load(os.path.join(DIR, "test_config/viz/viz_config_indices.yaml"))
        visualizer = Visualizer(config.visualization, mock_num_batches, batch_size, run_path)

        for epoch in range(epochs):
            run(5, visualizer, epoch, "train", data)
            run(5, visualizer, epoch, "test", data)
            run(5, visualizer, epoch, "val", data)

        self.assertEqual(len(os.listdir(os.path.join(run_path, "viz"))), 0)

    def test_indices(self):
        mock_data = Data()
        mock_data.pos = torch.zeros((batch_size, num_points, 3))
        mock_data.y = torch.zeros((batch_size, num_points, 1))
        mock_data.pred = torch.zeros((batch_size, num_points, 1))
        data = {"mock_date": mock_data}

        run_path = os.path.join(DIR, "test_viz")

        try:
            shutil.rmtree(run_path)
        except:
            pass
        if not os.path.exists(run_path):
            os.makedirs(run_path)

        mock_num_batches = {"train": 9, "test": 3, "val": 0}
        config = OmegaConf.load(os.path.join(DIR, "test_config/viz/viz_config_indices.yaml"))
        visualizer = Visualizer(config.visualization, mock_num_batches, batch_size, run_path)

        for epoch in range(epochs):
            run(5, visualizer, epoch, "train", data)
            run(5, visualizer, epoch, "test", data)
            run(5, visualizer, epoch, "val", data)

        targets = ["1_1.ply", "0_0.ply"]
        for split in ["train"]:
            for epoch in range(epochs):
                self.assertEqual(targets, os.listdir(os.path.join(run_path, "viz", str(epoch), split)))

        shutil.rmtree(run_path)

    def test_save_all(self):
        mock_data = Data()
        mock_data.pos = torch.zeros((num_points * batch_size, 3))
        mock_data.y = torch.zeros((num_points * batch_size, 1))
        mock_data.pred = torch.zeros((num_points * batch_size, 1))
        mock_data.batch = torch.zeros((num_points * batch_size))
        mock_data.batch[:num_points] = 1
        data = {"mock_date": mock_data}

        run_path = os.path.join(DIR, "test_viz")

        try:
            shutil.rmtree(run_path)
        except:
            pass
        if not os.path.exists(run_path):
            os.makedirs(run_path)

        epochs = 2
        num_samples = 100
        mock_num_batches = {"train": num_samples}

        config = OmegaConf.load(os.path.join(DIR, "test_config/viz/viz_config_save_all.yaml"))
        visualizer = Visualizer(config.visualization, mock_num_batches, batch_size, run_path)

        for epoch in range(epochs):
            run(num_samples // batch_size, visualizer, epoch, "train", data)

        for split in ["train"]:
            for epoch in range(epochs):
                current = set(os.listdir(os.path.join(run_path, "viz", str(epoch), split)))
                self.assertGreaterEqual(len(current), num_samples)

        shutil.rmtree(run_path)

    def test_pyg_data(self):
        mock_data = Data()
        mock_data.pos = torch.zeros((num_points * batch_size, 3))
        mock_data.y = torch.zeros((num_points * batch_size, 1))
        mock_data.pred = torch.zeros((num_points * batch_size, 1))
        mock_data.batch = torch.zeros((num_points * batch_size))
        mock_data.batch[:num_points] = 1
        data = {"mock_date": mock_data}

        run_path = os.path.join(DIR, "test_viz")

        try:
            shutil.rmtree(run_path)
        except:
            pass
        if not os.path.exists(run_path):
            os.makedirs(run_path)

        epochs = 50
        num_samples = 1000
        mock_num_batches = {"train": num_samples}

        config = OmegaConf.load(os.path.join(DIR, "test_config/viz/viz_config_non_deterministic.yaml"))
        visualizer = Visualizer(config.visualization, mock_num_batches, batch_size, run_path)

        for epoch in range(epochs):
            run(num_samples // batch_size, visualizer, epoch, "train", data)

        count = 0
        for split in ["train"]:
            target = set(os.listdir(os.path.join(run_path, "viz", "0", split)))
            for epoch in range(1, epochs):
                current = set(os.listdir(os.path.join(run_path, "viz", str(epoch), split)))
                count += 1 if len(target & current) == 0 else 0
        self.assertGreaterEqual(count, 49)

        shutil.rmtree(run_path)

    def test_dense_data(self):
        mock_data = Data()
        mock_data.pos = torch.zeros((batch_size, num_points, 3))
        mock_data.y = torch.zeros((batch_size, num_points, 1))
        mock_data.pred = torch.zeros((batch_size, num_points, 1))
        data = {"mock_date": mock_data}

        run_path = os.path.join(DIR, "test_viz")

        try:
            shutil.rmtree(run_path)
        except:
            pass
        if not os.path.exists(run_path):
            os.makedirs(run_path)

        mock_num_batches = {"train": 9, "test": 3, "val": 0}
        config = OmegaConf.load(os.path.join(DIR, "test_config/viz/viz_config_deterministic.yaml"))
        visualizer = Visualizer(config.visualization, mock_num_batches, batch_size, run_path)

        for epoch in range(epochs):
            run(5, visualizer, epoch, "train", data)
            run(5, visualizer, epoch, "test", data)
            run(5, visualizer, epoch, "val", data)

        for split in ["train", "test"]:
            targets = os.listdir(os.path.join(run_path, "viz", "0", split))
            for epoch in range(1, epochs):
                self.assertEqual(targets, os.listdir(os.path.join(run_path, "viz", str(epoch), split)))

        shutil.rmtree(run_path)


if __name__ == "__main__":
    unittest.main()