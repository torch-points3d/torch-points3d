import os
from glob import glob
from collections import defaultdict
import torch

from src.utils.colors import COLORS


def colored_print(color, msg):
    print(color + msg + COLORS.END_NO_TOKEN)


class ExperimentFolder:
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

    @property
    def contains_viz(self):
        if not hasattr(self, "_contains_viz"):
            for f in self._files:
                if "viz" in f:
                    self._viz_path = os.path.join(self._run_path, "viz")
                    self._contains_viz = True
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

    def from_split_to_file(self, event):
        if hasattr(self, "_current_epoch"):
            self._current_split = event.obj.value
            return self._current_experiment.get_files(self._current_epoch, self._current_split)
        else:
            return []

    def from_epoch_to_split(self, event):
        if hasattr(self, "_current_experiment"):
            self._current_epoch = event.obj.value
            return self._current_experiment.get_splits(self._current_epoch)
        else:
            return []

    def from_paths_to_epoch(self, event):
        run_path = event.obj.value
        for exp in self._current_exps:
            if str(run_path) == str(exp.__repr__()):
                self._current_experiment = exp
        return sorted(self._current_experiment.epochs)

    def get_model_wviz_paths(self, event):
        model_name = event.obj.value + ".pt"
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
