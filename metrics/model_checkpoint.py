import os
from os import path as osp
import tempfile
import warnings
import sys
import torch

from models.base_model import BaseModel
from metrics.colored_tqdm import COLORS
from utils_folder.utils import colored_print

DEFAULT_METRICS_FUNC = {
    "iou": max,
    "acc": max,
    "loss": min,
}  # Those map subsentences to their optimization functions


def get_model_checkpoint(
    model: BaseModel, log_dir: str = None, check_name: str = None, resume: bool = True, weight_name: str = None,
):

    model_checkpoint: ModelCheckpoint = ModelCheckpoint(log_dir, check_name, resume)

    if resume:
        model_checkpoint.initialize_model(model, weight_name)
    return model_checkpoint


class Checkpoint(object):
    def __init__(self, to_save: str = None, check_name: str = None, save_every_iter: bool = True):
        if not os.path.exists(to_save):
            os.makedirs(to_save)

        self._to_save = to_save
        self._check_name = check_name
        self._check_path = os.path.join(to_save, "{}.pt".format(check_name))
        self._initialize_objects()
        self._load_objects()

    def _initialize_objects(self):
        self._objects = {}
        self._objects["models"] = {}
        self._objects["stats"] = {"train": [], "test": [], "val": []}
        self._objects["optimizer"] = None
        self._objects["scheduler"] = None
        self._objects["args"] = None
        self._objects["kwargs"] = None
        self._filled = False

    def save_objects(self, models_to_save, stage, current_stat, optimizer, scheduler, **kwargs):
        self._objects["models"] = models_to_save
        self._objects["stats"][stage].append(current_stat)
        self._objects["optimizer"] = optimizer
        self._objects["scheduler"] = scheduler
        # self._objects['kwargs'] = kwargs
        torch.save(self._objects, self._check_path)

    def _load_objects(self):
        try:
            self._objects = torch.load(self._check_path)
            self._filled = True
        except:
            pass

    @property
    def models_to_save(self):
        return self._objects["models"]

    @property
    def stats(self):
        return self._objects["stats"]

    @property
    def is_empty(self):
        return not self._filled

    def get_optimizer(self):
        if not self.is_empty:
            try:
                return self._objects["optimizer"]
            except:
                raise Exception("The checkpoint doesn t contain an optimizer")

    def get_state_dict(self, weight_name):
        if not self.is_empty:
            try:
                models = self._objects["models"]
                try:
                    key_name = "best_{}".format(weight_name)
                    model = models[key_name]
                    print("Model loaded from {}:{}".format(self._check_path, key_name))
                    return model
                except:
                    key_name = "default"
                    model = models["default"]
                    print("Model loaded from {}:{}".format(self._check_path, key_name))
                    return model
            except:
                raise Exception("This weight name isn't within the checkpoint ")

    @staticmethod
    def load_objects(to_save: str = None, check_name: str = None):
        return Checkpoint(to_save, check_name)


class ModelCheckpoint(object):
    def __init__(self, to_save: str = None, check_name: str = None, resume: bool = True):
        self._checkpoint = Checkpoint.load_objects(to_save, check_name)
        self._resume = resume

    @property
    def start_epoch(self):
        if self._resume:
            return self.get_starting_epoch()
        else:
            return 1

    def get_starting_epoch(self):
        return len(self._checkpoint.stats["train"]) + 1

    def initialize_model(self, model: BaseModel, weight_name: str = None):
        if not self._checkpoint.is_empty:
            state_dict = self._checkpoint.get_state_dict(weight_name)
            model.load_state_dict(state_dict)
            optimizer = self._checkpoint.get_optimizer()
            model.set_optimizer(optimizer.__class__, lr=optimizer.defaults["lr"])

    def find_func_from_metric_name(self, metric_name, default_metrics_func):
        for token_name, func in default_metrics_func.items():
            if token_name in metric_name:
                return func
        raise Exception(
            'The metric name doesn t have a func to measure which one is best. Example: For best_train_iou, {"iou":max}'
        )

    def save_best_models_under_current_metrics(self, model: BaseModel, metrics_holder: dict, **kwargs):
        """[This function is responsible to save checkpoint under the current metrics and their associated DEFAULT_METRICS_FUNC]

        Arguments:
            model {[BaseModel]} -- [Model]
            metrics_holder {[Dict]} -- [Need to contain stage, epoch, current_metrics]
        """

        metrics = metrics_holder["current_metrics"]
        stage = metrics_holder["stage"]
        epoch = metrics_holder["epoch"]

        stats = self._checkpoint.stats
        state_dict = model.state_dict()
        optimizer = model.optimizer

        current_stat = {}
        current_stat["epoch"] = epoch

        models_to_save = self._checkpoint.models_to_save

        if stage == "train":
            models_to_save["default"] = state_dict

        if len(stats[stage]) > 0:
            latest_stats = stats[stage][-1]

            msg = ""
            improved_metric = 0

            for metric_name, current_metric_value in metrics.items():
                current_stat[metric_name] = current_metric_value

                metric_func = self.find_func_from_metric_name(metric_name, DEFAULT_METRICS_FUNC)
                best_metric_from_stats = latest_stats["best_{}".format(metric_name)]
                best_value = metric_func(best_metric_from_stats, current_metric_value)
                current_stat["best_{}".format(metric_name)] = best_value

                # This new value seems to be better under metric_func
                if ("test" == stage) and (current_metric_value == best_value):  # Update the model weights
                    models_to_save["best_{}".format(metric_name)] = state_dict

                    msg += "{}: {} -> {}, ".format(metric_name, best_metric_from_stats, best_value)
                    improved_metric += 1

            if improved_metric > 0:
                colored_print(COLORS.VAL_COLOR, msg[:-2])
        else:
            # stats[stage] is empty.
            for metric_name, metric_value in metrics.items():
                current_stat[metric_name] = metric_value
                current_stat["best_{}".format(metric_name)] = metric_value

        self._checkpoint.save_objects(models_to_save, stage, current_stat, optimizer, model.scheduler, **kwargs)
