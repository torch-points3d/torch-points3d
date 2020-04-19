import os
import sys
import argparse
from glob import glob
from collections import defaultdict
import torch
import shutil

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

from torch_points3d.utils.colors import COLORS


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

    @property
    def stats(self):
        return self._stats


def main(args):

    experiment_with_models = defaultdict(list)
    run_paths = glob(os.path.join(ROOT, "outputs", "*", "*"))
    for run_path in run_paths:
        experiment = ExperimentFolder(run_path)
        if experiment.contains_trained_model:
            experiment_with_models[experiment.model_name].append(experiment)
        else:
            if args.d:
                shutil.rmtree(run_path)

    print("")
    for model_name in experiment_with_models.keys():
        colored_print(COLORS.Green, str(model_name))
        for experiment in experiment_with_models[model_name]:
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

    if args.pdb:
        import pdb

        pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find experiments")
    parser.add_argument("-d", action="store_true", default=False, help="Delete empty folders")
    parser.add_argument("-pdb", action="store_true", default=False, help="Activate pdb for explore Experiment Folder")
    args = parser.parse_args()
    main(args)
