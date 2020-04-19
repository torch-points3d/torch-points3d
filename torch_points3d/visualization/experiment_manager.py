import os
from glob import glob
from collections import defaultdict
import torch
from plyfile import PlyData, PlyElement
from numpy.lib import recfunctions as rfn
from torch_points3d.utils.colors import COLORS
import numpy as np


def colored_print(color, msg):
    print(color + msg + COLORS.END_NO_TOKEN)


class ExperimentFolder:

    POS_KEYS = ["x", "y", "z"]

    def __init__(self, run_path):
        self._run_path = run_path
        self._model_name = None
        self._stats = None
        self._find_files()

    def _find_files(self):
        self._files = os.listdir(self._run_path)

    def __repr__(self):
        return self._run_path.split("outputs")[1]

    @property
    def model_name(self):
        return self._model_name

    @property
    def epochs(self):
        return os.listdir(self._viz_path)

    def get_splits(self, epoch):
        return os.listdir(os.path.join(self._viz_path, str(epoch)))

    def get_files(self, epoch, split):
        return os.listdir(os.path.join(self._viz_path, str(epoch), split))

    def load_ply(self, epoch, split, file):
        self._data_name = "data_{}_{}_{}".format(epoch, split, file)
        if not hasattr(self, self._data_name):
            path_to_ply = os.path.join(self._viz_path, str(epoch), split, file)
            if os.path.exists(path_to_ply):
                plydata = PlyData.read(path_to_ply)
                arr = np.asarray([e.data for e in plydata.elements])
                names = list(arr.dtype.names)
                pos_indices = [names.index(n) for n in self.POS_KEYS]
                non_pos_indices = {n: names.index(n) for n in names if n not in self.POS_KEYS}
                arr_ = rfn.structured_to_unstructured(arr).squeeze()
                xyz = arr_[:, pos_indices]
                data = {"xyz": xyz, "columns": non_pos_indices.keys(), "name": self._data_name}
                for n, i in non_pos_indices.items():
                    data[n] = arr_[:, i]
                setattr(self, self._data_name, data)
            else:
                print("The file doesn' t exist: Wierd !")
        else:
            return getattr(self, self._data_name)

    @property
    def current_pointcloud(self):
        return getattr(self, self._data_name)

    @property
    def contains_viz(self):
        if not hasattr(self, "_contains_viz"):
            for f in self._files:
                if "viz" in f:
                    self._viz_path = os.path.join(self._run_path, "viz")
                    vizs = os.listdir(self._viz_path)
                    self._contains_viz = len(vizs) > 0
                    return self._contains_viz
            self._contains_viz = False
            return self._contains_viz
        else:
            return self._contains_viz

    @property
    def contains_trained_model(self):
        if not hasattr(self, "_contains_trained_model"):
            for f in self._files:
                if ".pt" in f:
                    self._contains_trained_model = True
                    self._model_name = f
                    return self._contains_trained_model
            self._contains_trained_model = False
            return self._contains_trained_model
        else:
            return self._contains_trained_model

    def extract_stats(self):
        path_to_checkpoint = os.path.join(self._run_path, self.model_name)
        stats = torch.load(path_to_checkpoint)["stats"]
        self._stats = stats
        num_epoch = len(stats["train"])
        stats_dict = defaultdict(dict)
        for split_name in stats.keys():
            if len(stats[split_name]) > 0:
                latest_epoch = stats[split_name][-1]
                for metric_name in latest_epoch.keys():
                    if "best" in metric_name:
                        stats_dict[metric_name][split_name] = latest_epoch[metric_name]
        return num_epoch, stats_dict


class ExperimentManager(object):
    def __init__(self, experiments_root):
        self._experiments_root = experiments_root
        self._collect_experiments()

    def _collect_experiments(self):
        self._experiment_with_models = defaultdict(list)
        run_paths = glob(os.path.join(self._experiments_root, "outputs", "*", "*"))
        for run_path in run_paths:
            experiment = ExperimentFolder(run_path)
            if experiment.contains_trained_model:
                self._experiment_with_models[experiment.model_name].append(experiment)

        self._find_experiments_with_viz()

    def _find_experiments_with_viz(self):
        if not hasattr(self, "_experiment_with_viz"):
            self._experiment_with_viz = defaultdict(list)
            for model_name in self._experiment_with_models.keys():
                for experiment in self._experiment_with_models[model_name]:
                    if experiment.contains_viz:
                        self._experiment_with_viz[experiment.model_name].append(experiment)

    @property
    def model_name_wviz(self):
        keys = list(self._experiment_with_viz.keys())
        return [k.replace(".pt", "") for k in keys]

    @property
    def current_pointcloud(self):
        return self._current_experiment.current_pointcloud

    def load_ply_file(self, file):
        if hasattr(self, "_current_split"):
            self._current_file = file
            self._current_experiment.load_ply(self._current_epoch, self._current_split, self._current_file)
        else:
            return []

    def from_split_to_file(self, split_name):
        if hasattr(self, "_current_epoch"):
            self._current_split = split_name
            return self._current_experiment.get_files(self._current_epoch, self._current_split)
        else:
            return []

    def from_epoch_to_split(self, epoch):
        if hasattr(self, "_current_experiment"):
            self._current_epoch = epoch
            return self._current_experiment.get_splits(self._current_epoch)
        else:
            return []

    def from_paths_to_epoch(self, run_path):
        for exp in self._current_exps:
            if str(run_path) == str(exp.__repr__()):
                self._current_experiment = exp
        return sorted(self._current_experiment.epochs)

    def get_model_wviz_paths(self, model_path):
        model_name = model_path + ".pt"
        self._current_exps = self._experiment_with_viz[model_name]
        return self._current_exps

    def display_stats(self):
        print("")
        for model_name in self._experiment_with_models.keys():
            colored_print(COLORS.Green, str(model_name))
            for experiment in self._experiment_with_models[model_name]:
                print(experiment)
                num_epoch, stats = experiment.extract_stats()
                colored_print(COLORS.Red, "Epoch: {}".format(num_epoch))
                for metric_name in stats:
                    sentence = ""
                    for split_name in stats[metric_name].keys():
                        sentence += "{}: {}, ".format(split_name, stats[metric_name][split_name])
                    metric_sentence = metric_name + "({})".format(sentence[:-2])
                    colored_print(COLORS.BBlue, metric_sentence)
                print("")
            print("")
