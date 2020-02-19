import os
import sys
import argparse
from glob import glob
from collections import defaultdict

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)


class ExperimentFolder:
    def __init__(self, run_path):
        self._run_path = run_path

    def _find_info(self):
        self._files = os.listdir(self._run_path)

    def __repr__(self):
        return self._run_path.split("outputs")[1]


def get_pt_file(files):
    for f in files:
        if ".pt" in f:
            return f
    return None


def filter_on_saved_models(run_path):
    files = os.listdir(run_path)
    return get_pt_file(files)


def gather_info(run_path):
    exp = ExperimentFolder(run_path)
    return exp


def main(args):

    trained_models = defaultdict(list)
    run_paths = glob(os.path.join(ROOT, "outputs", "*", "*"))
    for run_path in run_paths:
        trained_model = filter_on_saved_models(run_path)
        if trained_model:
            trained_models[trained_model].append(gather_info(run_path))

    print()
    for model_name in trained_models.keys():
        print(model_name)
        for path in trained_models[model_name]:
            print(path)
        print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Find experiments")
    parser.add_argument("-e", type=str, help="Gather epochs training")
    parser.add_argument("-s", type=str, help="Gather stats")
    args = parser.parse_args()
    main(args)
